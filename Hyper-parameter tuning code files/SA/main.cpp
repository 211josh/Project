///Simulated Annealing - tuning hyper-parameters via random search

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sqlite3.h>
#include <tgmath.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <random>

#include "bitboard.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"
#include "evaluate.h"
#include "gnuplot-iostream.h"

const int no_features = 36; //no. features.'

///FINDING BEST HYPER-PARAMETERS
const int no_runs = 250;

///HYPER-PARAMETERS
const int no_positions = 500;
const int min_no_generations = 3500, max_no_generations = 6000;
const float min_startTEMP = 750.0, max_startTEMP = 1250.0;
const float min_coolingRate = 0.999, max_coolingRate = 0.9999;
const float min_endTemp = 0.05, max_endTemp = 0.2;
const int min_weight_range = 10, max_weight_range = 50;
const int min_weight_change_max = 1, max_weight_change_max=10;
const int min_no_weight_change_max = 1, max_no_weight_change_max=10;

///Best hyper-parameters
struct HP {
    int no_generations;
    float startTEMP;
    float coolingRate;
    float endTemp;
    int weight_range;
    int weight_change_max;
    int no_weight_change_max;
    float best_fitness;
};
HP bestHP; //best HP stored throughout.

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> distrib_generations(min_no_generations, max_no_generations);
std::uniform_real_distribution<float> distrib_startTEMP(min_startTEMP, max_startTEMP);
std::uniform_real_distribution<float> distrib_coolingRate(min_coolingRate, max_coolingRate);
std::uniform_real_distribution<float> distrib_endTemp(min_endTemp, max_endTemp);
std::uniform_int_distribution<> distrib_weight_range(min_weight_range, max_weight_range);
std::uniform_int_distribution<> distrib_weight_change_max(min_weight_change_max, max_no_weight_change_max);
std::uniform_int_distribution<> distrib_no_weight_change_max(min_no_weight_change_max, max_no_weight_change_max);


int k = 0;
int eval_temp;
float eval_temp_float;
float current_eval_difference;
float new_eval_difference;
int current_sample[no_features];

std::vector<int> current_weights(no_features); //used to calculate evaluation function of each sample.
std::vector<int> graph_generation; //x-axis of graph.
std::vector<float> graph_eval_dif; //used in graph. fitness function at each generation of best sample in centipawns.

std::vector<std::string> fen4850(4850); //stores fens from database.
std::vector<int> evalDB4850(4850); //stores static_evals from database.
std::vector<int> randomPositions; //random positions selected from database for fitness function at each generation.

using namespace Stockfish;

///accessDB stores relevant data from our database in vectors at the beginning of the algorithm.
void accessDB(){
    sqlite3 *db;
    sqlite3_stmt *stmt;
    if (sqlite3_open("fen_depth2.db", &db) != SQLITE_OK) {
        return;
    }
    const char *sql = "SELECT id, fen, static_eval FROM evaluations LIMIT 4850";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) != SQLITE_OK) {
        return;
    }
    int i = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        fen4850[i] = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        evalDB4850[i] = sqlite3_column_double(stmt, 2);
        ++i;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}


int main(int argc, char* argv[]) {
  Gnuplot gp;
  accessDB();

  bestHP.best_fitness = 10000000;

  ///Initialisation required for evaluation function to be functioning.
  std::cout << engine_info() << std::endl;
  UCI::init(Options);
  Tune::init();
  Bitboards::init();
  Position::init();
  Threads.set(size_t(Options["Threads"]));
  Search::clear();

  //FEN initialisation
  std::string fen = "rn1qkbnr/ppp2ppp/8/3pp3/4P1b1/5N2/PPPP1PPP/RNB1KB1R w KQkq - 0 4"; //example fen.
  StateListPtr states(new std::deque<StateInfo>(1));
  Position pos;
  pos.set(fen, false, &states->back(), Threads.main());

  for(int t=0; t < no_runs; ++t){
      graph_generation.push_back(t);
      std::cout << t << '\n';

      ///Random hyper-parameters within ranges
      int no_generations = distrib_generations(gen);
      float startTEMP = distrib_startTEMP(gen);
      float coolingRate = distrib_coolingRate(gen);
      float endTemp = distrib_endTemp(gen);
      int weight_range = distrib_weight_range(gen);
      int weight_change_max = distrib_weight_range(gen);
      int no_weight_change_max = distrib_weight_change_max(gen);


      ///Initialise first sample:
      for (int i = 0; i < no_features; ++i){
        int random_weight = std::rand() % weight_range + 1; // generate random weight between 1 and (weight_range)
        current_sample[i] = random_weight; // assign random weight i to the sample.
      }

      ///MAIN SIMULATED ANNEALING ALGORITHM LOOP
      float temperature = startTEMP;
      int k = 0;
      while(temperature > endTemp && k < no_generations){ //until either stopping criteria met:
        k += 1;
        current_eval_difference = 0;
        new_eval_difference = 0;
        //std::cout << "GENERATION: " << '\n';
        //std::cout << k << '\n';
        //std::cout << "TEMPERATURE: " << '\n';
        //std::cout << temperature << '\n';


        ///RANDOM SELECTION OF POSITIONS FOR FITNESS FUNCTION
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 4849);
        for (int i = 0; i < no_positions; ++i) {
            int randomNumber = distrib(gen); //random number generated uniformly on interval [0,4849]
            randomPositions.push_back(randomNumber);
        }

        ///EVALUATE FITNESS OF CURRENT SAMPLE
        for (int i = 0; i < no_features; ++i){
          current_weights[i] = current_sample[i]; //set current_weights to sample weights for evaluation function.
        }
        for(int n=0; n < no_positions; ++n){
            //set fen.
            fen = fen4850[randomPositions[n]];
            StateListPtr states(new std::deque<StateInfo>(1));
            pos.set(fen, false, &states->back(), Threads.main());
            //evaluate.
            Value v = Eval::evaluate(pos);
            eval_temp = int(v)*100/PawnValueEg;
            current_eval_difference += fabs(eval_temp - evalDB4850[randomPositions[n]]); //overall difference in centipawns across positions.
        }

        ///Generate a new sample
        int new_sample[no_features];
        std::copy(current_sample, current_sample + no_features, new_sample);  //copy current sample to new one.

        int num_weights_to_alter = std::rand() % no_weight_change_max + 1;  //random no. weights to alter, selected uniformly between 1 and (no_weight_change_max).
        for (int i = 0; i < num_weights_to_alter; ++i) {
            int index_to_alter = std::rand() % no_features;  //choose random weight to alter.
            float neighbour_weight_dif = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (2 * weight_change_max) - weight_change_max;
            new_sample[index_to_alter] += neighbour_weight_dif;  //change weight.
            //boundary checks to ensure the weight stays within [1, weight_range].
            if (new_sample[index_to_alter] < 1) {
            new_sample[index_to_alter] = 1;
            } else if (new_sample[index_to_alter] > weight_range) {
            new_sample[index_to_alter] = weight_range;
            }
        }

        ///Evaluate new sample
        for (int i = 0; i < no_features; ++i) {
            current_weights[i] = new_sample[i];
        }
        for(int n=0; n < no_positions; ++n){
            //SET FEN
            fen = fen4850[randomPositions[n]];
            StateListPtr states(new std::deque<StateInfo>(1));
            pos.set(fen, false, &states->back(), Threads.main());

            //EVALUATE
            Value v = Eval::evaluate(pos);
            eval_temp = int(v)*100/PawnValueEg;
            new_eval_difference += fabs(eval_temp - evalDB4850[randomPositions[n]]); //overall difference in centipawns across positions.
        }

        ///Decide if our new sample is our old sample
        float random_num = ((double) rand() / (RAND_MAX)); //[0,1)
        if (new_eval_difference < current_eval_difference || random_num < (1/(1+exp(((new_eval_difference-current_eval_difference)/temperature))))) {
            current_eval_difference = new_eval_difference;
            std::copy(new_sample, new_sample + no_features, current_sample);
        }

        //Reduce the temperature
        temperature *= coolingRate;
      }
  if((current_eval_difference/no_positions)<bestHP.best_fitness){
            bestHP.no_generations = no_generations;
            bestHP.startTEMP = startTEMP;
            bestHP.coolingRate = coolingRate;
            bestHP.endTemp = endTemp;
            bestHP.weight_range = weight_range;
            bestHP.weight_change_max = weight_change_max;
            bestHP.no_weight_change_max = no_weight_change_max;
            bestHP.best_fitness = (current_eval_difference/no_positions);
    }

    graph_eval_dif.push_back(bestHP.best_fitness);
    std::cout << "BEST FITNESS VALUE:" << '\n';
    std::cout << bestHP.best_fitness << '\n';
  }

  ///Graphing of performance
  gp << "set xlabel 'Algorithm run'\n";
  gp << "set ylabel 'Average evaluation difference over positions (centipawns)'\n";
  gp << "set title 'Fitness of best sample found over hyper-parameter tuning'\n";
  gp << "set yrange [0:*]\n";
  std::vector<std::pair<int,float>> plot_data;
  for(size_t i = 0; i < graph_generation.size(); ++i){
    plot_data.push_back(std::make_pair(graph_generation[i], graph_eval_dif[i]));
  }
  gp << "plot '-' with lines title 'Best Eval Diff'\n";
  gp.send1d(plot_data);

  ///BEST HYPER-PARAMETERS FOUND:
  std::cout << "bestHP.no_generations: " << '\n';
  std::cout << bestHP.no_generations << '\n';

  std::cout << "bestHP.startTEMP: " << '\n';
  std::cout << bestHP.startTEMP << '\n';

  std::cout << "bestHP.coolingRate: " << '\n';
  std::cout << bestHP.coolingRate << '\n';

  std::cout << "bestHP.endTemp: " << '\n';
  std::cout << bestHP.endTemp << '\n';

  std::cout << "bestHP.weight_range: " << '\n';
  std::cout << bestHP.weight_range << '\n';

  std::cout << "bestHP.weight_change_max: " << '\n';
  std::cout << bestHP.weight_change_max << '\n';

  std::cout << "bestHP.no_weight_change_max: " << '\n';
  std::cout << bestHP.no_weight_change_max << '\n';

  std::cout << "Fitness of best sample using hyper-parameters: " << '\n';
  std::cout << bestHP.best_fitness << '\n';


  return 0;
}
