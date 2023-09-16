///Simulated Annealing to tune the parameters of our evaluation function features.

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

const int no_features = 36; //no. features.

///HYPER-PARAMETERS
const int no_positions = 500; //no. positions in fitness evaluation. randomly selected each generation out of 4850 possible positions.
const int no_generations = 5000; //max no. iterations of algorithm.
const float startTEMP = 1000.0; //starting temp.
const float coolingRate = 0.9995; //cooling rate at each generation.
const float endTemp = 0.1; //lowest temperature of algorithm - stopping criteria.
const int weight_range = 30; //range for which weights can take value on interval [1,weight_range].
const int weight_change_max = 6; //max number which each weight can change by. weight changed selected uniformly on [-weight_change_max,weight_change_max].
const int no_weight_change_max = 5; //max no. weights to be slightly altered for neighbour sample. randomly selected on interval [1 - no_weight_change_max].


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

  std::srand(std::time(nullptr)); //makes randomness completely random each run. remove for consistent randomness.

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

  ///Initialise first sample:
  for (int i = 0; i < no_features; ++i){
    int random_weight = std::rand() % weight_range + 1; // generate random weight between 1 and (weight_range)
    current_sample[i] = random_weight; // assign random weight i to the sample.
  }

  ///MAIN SIMULATED ANNEALING ALGORITHM LOOP
  float temperature = startTEMP;

  while(temperature > endTemp && k < no_generations){ //until either stopping criteria met:
    k += 1;
    current_eval_difference = 0;
    new_eval_difference = 0;
    std::cout << "GENERATION: " << '\n';
    std::cout << k << '\n';
    std::cout << "TEMPERATURE: " << '\n';
    std::cout << temperature << '\n';

    graph_generation.push_back(k);

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

    graph_eval_dif.push_back(current_eval_difference/no_positions);

    //Reduce the temperature
    temperature *= coolingRate;
  }

  ///Graphing of performance
  gp << "set xlabel 'Generation'\n";
  gp << "set ylabel 'Fitness in centipawns'\n";
  gp << "set title 'Fitness of best sample'\n";
  gp << "set yrange [0:*]\n";
  std::vector<std::pair<int,float>> plot_data;
  for(size_t i = 0; i < graph_generation.size(); ++i){
    plot_data.push_back(std::make_pair(graph_generation[i], graph_eval_dif[i]));
  }

  std::cout << "CURRENT EVAL DIF: " << '\n';
  std::cout << current_eval_difference << '\n';
  std::cout << "GEN BACK: " << '\n';
  std::cout << graph_generation.back() << '\n';

  int final_ = static_cast<int>(current_eval_difference/no_positions);

  gp << "set label ' " << final_ << "' at " << graph_generation.back() << "," << final_ << " point pointtype 7 offset char -5,-1\n";
  gp << "plot '-' with lines title 'Fitness of best sample'\n";
  gp.send1d(plot_data);

  ///Weights of best sample when algorithm complete.
  std::cout << "bishopPairWeight: " << '\n';
  std::cout << current_sample[0] << '\n';

  std::cout << "shelterWeight: " << '\n';
  std::cout << current_sample[1] << '\n';

  std::cout << "pawnStormWeight: " << '\n';
  std::cout << current_sample[2] << '\n';

  std::cout << "kingEscapeWeight: " << '\n';
  std::cout << current_sample[3] << '\n';

  std::cout << "kingTropismWeight: " << '\n';
  std::cout << current_sample[4] << '\n';

  std::cout << "doubledPWeight: " << '\n';
  std::cout << current_sample[5] << '\n';

  std::cout << "isolatedPWeight: " << '\n';
  std::cout << current_sample[6] << '\n';

  std::cout << "PPWeight: " << '\n';
  std::cout << current_sample[7] << '\n';

  std::cout << "rooksBehindWeight: " << '\n';
  std::cout << current_sample[8] << '\n';

  std::cout << "backwardPWeight: " << '\n';
  std::cout << current_sample[9] << '\n';

  std::cout << "chainsWeight: " << '\n';
  std::cout << current_sample[10] << '\n';

  std::cout << "pawnMobWeight: " << '\n';
  std::cout << current_sample[11] << '\n';

  std::cout << "phalanxWeight: " << '\n';
  std::cout << current_sample[12] << '\n';

  std::cout << "outpostsWeight: " << '\n';
  std::cout << current_sample[13] << '\n';

  std::cout << "knightMobWeight: " << '\n';
  std::cout << current_sample[14] << '\n';

  std::cout << "bishopMobWeight: " << '\n';
  std::cout << current_sample[15] << '\n';

  std::cout << "rookMobWeight: " << '\n';
  std::cout << current_sample[16] << '\n';

  std::cout << "queenMobWeight: " << '\n';
  std::cout << current_sample[17] << '\n';

  std::cout << "ccWeight: " << '\n';
  std::cout << current_sample[18] << '\n';

  std::cout << "rookFileWeight: " << '\n';
  std::cout << current_sample[19] << '\n';

  std::cout << "queenFileWeight: " << '\n';
  std::cout << current_sample[20] << '\n';

  std::cout << "trapBishWeight: " << '\n';
  std::cout << current_sample[21] << '\n';

  std::cout << "trapRookWeight: " << '\n';
  std::cout << current_sample[22] << '\n';

  std::cout << "pinsWeight: " << '\n';
  std::cout << current_sample[23] << '\n';

  std::cout << "forksWeight: " << '\n';
  std::cout << current_sample[24] << '\n';

  std::cout << "skewersWeight: " << '\n';
  std::cout << current_sample[25] << '\n';

  std::cout << "discAttWeight: " << '\n';
  std::cout << current_sample[26] << '\n';

  std::cout << "doubAttWeight: " << '\n';
  std::cout << current_sample[27] << '\n';

  std::cout << "overloadWeight: " << '\n';
  std::cout << current_sample[28] << '\n';

  std::cout << "hangingWeight: " << '\n';
  std::cout << current_sample[29] << '\n';

  std::cout << "egkingActWeight: " << '\n';
  std::cout << current_sample[30] << '\n';

  std::cout << "rookCoordWeight: " << '\n';
  std::cout << current_sample[31] << '\n';

  std::cout << "otherCoordWeight: " << '\n';
  std::cout << current_sample[32] << '\n';

  std::cout << "kingCoordWeight: " << '\n';
  std::cout << current_sample[33] << '\n';

  std::cout << "fiancettoWeight: " << '\n';
  std::cout << current_sample[34] << '\n';

  std::cout << "developWeight: " << '\n';
  std::cout << current_sample[35] << '\n';

  return 0;
}
