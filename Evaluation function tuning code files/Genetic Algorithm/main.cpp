///Genetic algorithm to tune the parameters of our evaluation function features.

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sqlite3.h>
#include <tgmath.h>
#include <cstdio>
#include <ctime>
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

const int no_features = 36; //no. features in evaluation function.

///HYPER-PARAMETERS
const int no_samples = 100; //no. samples at each generation.
const int no_positions = 500; //no. positions in fitness evaluation. randomly selected each generation out of 4850 possible positions.
const int no_generations = 100; //max no. iterations of algorithm.
const int select_percent = 10; //percentage out of 100 of samples selected for next generation.
const float mutRate = 0.20; //chance of a weight in a sample being mutated.
const float mutRange = 6; //interval of which mutations can effect weights. uniformly selected on interval [-mutRange,mutRange].
const int weight_range = 30; //range for which weights can take value on interval [1,weight_range].

int eval_temp;
const int no_saved = static_cast<int>(no_samples * (static_cast<float>(select_percent) / 100.0));
int samples[no_samples][no_features]; //(no_samples) samples, each with one weight for each feature.
int selected_samples[no_saved]; //samples which are selected.

std::vector<int> current_weights(no_features); //weights used in evaluation function, updated for each sample.
std::vector<int> eval_difference(no_samples); //fitness function for each sample, updated each generation.
std::vector<float> graph_eval_dif; //used in graph. fitness function at each generation of best sample in centipawns.
std::vector<int> graph_generation; //x-axis of graph.
std::vector<int> graph_eval_dif_average;

std::vector<std::string> fen4850(4850); //stores fens from database. [0 - 4849]
std::vector<int> evalDB4850(4850); //stores static_evals from database [0 - 4849]
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

///Crossover functions. We include 3 different types of crossover functions for variety.
//One-point crossover
void onePointCrossover(const int parent1[], const int parent2[], int child[], const int size) {
    int crossover_point = std::rand() % size;
    for (int i = 0; i < size; ++i) {
        if (i < crossover_point) {
            child[i] = parent1[i];
        } else {
            child[i] = parent2[i];
        }
    }
}
//Two-point crossover
void twoPointCrossover(const int parent1[], const int parent2[], int child[], const int size) {
    int crossover_point1 = std::rand() % size;
    int crossover_point2 = std::rand() % size;
    if (crossover_point1 > crossover_point2) std::swap(crossover_point1, crossover_point2);
    for (int i = 0; i < size; ++i) {
        if (i < crossover_point1 || i > crossover_point2) {
            child[i] = parent1[i];
        } else {
            child[i] = parent2[i];
        }
    }
}
//Uniform crossover
void uniformCrossover(const int parent1[], const int parent2[], int child[], const int size) {
    for (int i = 0; i < size; ++i) {
        child[i] = (std::rand() % 2) ? parent1[i] : parent2[i];
    }
}

int main(int argc, char* argv[]) {
  std::srand(std::time(nullptr)); //makes randomness completely random each run. remove for consistent randomness.
  Gnuplot gp;
  accessDB();

  ///Initialise first population of samples (weights):
  for (int i = 0; i < no_samples; ++i){
    for (int j = 0; j < no_features; ++j){
        int random_weight = std::rand() % weight_range + 1; //generate random weight between 1 and (weight_range).
        samples[i][j] = random_weight; //assign random weight j to sample i.
    }
  }

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

///commented out code below is for the fitness function which used a depth of 2.
//  //Depth to search
//  Depth depth = 2;
//
//  //Perform search
//  Search::LimitsType limits;
//  limits.depth = depth;
//  Threads.start_thinking(pos, states, limits);
//  Threads.main()->wait_for_search_finished();
//
//  //Get the evaluation
//  if (!Threads.main()->rootMoves.empty()) {
//    Value v = Threads.main()->rootMoves[0].score;
//    int cp_value = int(v) * 100 / PawnValueEg;
//  //    std::cout << "Final evaluation in centipawns: " << cp_value << std::endl;
//  ]

  ///Main genetic algorithm loop
  for (int k = 0; k<no_generations; ++k){
    randomPositions.clear();
    graph_generation.push_back(k+1); //add generation number to graph.
    int eval_dif_total = 0;
    ///RANDOM SELECTION OF POSITIONS FOR FITNESS FUNCTION
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 4849);
    for (int i = 0; i < no_positions; ++i) {
        int randomNumber = distrib(gen); //random number generated uniformly on interval [0,4849]
        randomPositions.push_back(randomNumber);
    }

    ///FITNESS EVALUATION
    std::cout << "GENERATION: " << '\n';
    std::cout << k << '\n';
    for (int i = 0; i < no_samples; ++i){
        eval_difference[i] = 0;
        for (int j = 0; j < no_features; ++j){
            current_weights[j] = samples[i][j]; //set current_weights to sample weights for evaluation function.
        }
        for(int n=0; n < no_positions; ++n){ //for each position:
//            Search::clear(); //Only need for depth search. Very slow.
            fen = fen4850[randomPositions[n]];
            StateListPtr states(new std::deque<StateInfo>(1));
            pos.set(fen, false, &states->back(), Threads.main());
            ///Below is for search of depth 2. Too slow.
//            Search::LimitsType limits;
//            limits.depth = depth;
//            Threads.start_thinking(pos, states, limits);
//            Threads.main()->wait_for_search_finished();
//            if (!Threads.main()->rootMoves.empty()) {
//                Value v = Threads.main()->rootMoves[0].score;
//                eval_temp = int(v)*100/PawnValueEg;
//		  eval_difference[i] += fabs(eval_temp - evalDB4850[randomPositions[n]]);
//            }
            ///Static-eval
            Value v = Eval::evaluate(pos);
            eval_temp = int(v)*100/PawnValueEg;
            eval_difference[i] += fabs(eval_temp - evalDB4850[randomPositions[n]]); //overall difference in centipawns across positions.
        }
        eval_dif_total += eval_difference[i];
    }
    graph_eval_dif_average.push_back((eval_dif_total/no_samples)/no_positions);

    ///SELECTION
    std::vector<std::pair<float, int>> evalWithIndex;  //selecting best (select_percent)% samples based on smallest overall eval_difference.
    for (int i = 0; i < no_samples; ++i) {
        evalWithIndex.push_back(std::make_pair(eval_difference[i], i));
    }
    std::sort(evalWithIndex.begin(), evalWithIndex.end(),[](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    });
    for (int i = 0; i < no_saved; ++i){
        selected_samples[i] = evalWithIndex[i].second; //store selected samples via their index in samples.
    }
    graph_eval_dif.push_back((eval_difference[selected_samples[0]])/no_positions); //averaged out over number of positions in graph.

    ///CROSSOVER
    for (int i = no_saved; i < no_samples; ++i) {  //crossover selected samples until population back up to (no_samples).
      int parent1_idx = selected_samples[std::rand() % no_saved];
      int parent2_idx = selected_samples[std::rand() % no_saved]; //randomly select two parents from the selected_samples.
      int *parent1 = samples[parent1_idx];
      int *parent2 = samples[parent2_idx];
      int crossover_method = std::rand() % 3; //equal chance of each 3 types of crossover being performed.
      if (crossover_method == 0) {
        onePointCrossover(parent1, parent2, samples[i], no_features);
      }
      else if (crossover_method == 1) {
      twoPointCrossover(parent1, parent2, samples[i], no_features);
      }
      else {
        uniformCrossover(parent1, parent2, samples[i], no_features);
      }
    }

    ///MUTATION
    for (int i = 0; i < no_samples; ++i) { //for each new child sample:
      if (std::find(std::begin(selected_samples), std::end(selected_samples), i) != std::end(selected_samples)) {
        continue; //skip the mutation for samples that were selected in the previous generation
      }
      for (int j = 0; j < no_features; ++j) { //for each weight:
          float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX); //random float between 0 and 1.
          if (r < mutRate) { //if less than mutation rate, mutate this weight.
              float mutSize = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (2 * mutRange) - mutRange; //random value between [-mutRange, mutRange].
              samples[i][j] += mutSize;
              //ensure new value stays in the range [1, weight_range]
            if (samples[i][j] > weight_range) {
                samples[i][j] = weight_range;
            }
              if (samples[i][j] < 1) {
                  samples[i][j] = 1;
              }
          }
      }
    }
  }

  ///Graphing of performance
  gp << "set xlabel 'Generation'\n";
  gp << "set ylabel 'Fitness in centipawns'\n";
  gp << "set title 'Fitness of best sample and population average at each generation'\n";
  gp << "set yrange [0:*]\n";
  std::vector<std::pair<int,float>> plot_data;
  std::vector<std::pair<int,float>> plot_data_average;
  for(size_t i = 0; i < graph_generation.size(); ++i){
    plot_data.push_back(std::make_pair(graph_generation[i], graph_eval_dif[i]));
    plot_data_average.push_back(std::make_pair(graph_generation[i], graph_eval_dif_average[i]));
  }
  gp << "set label ' " << (eval_difference[selected_samples[0]])/no_positions << "' at " << graph_generation.back() << "," << (eval_difference[selected_samples[0]])/no_positions << " point pointtype 7 offset char -5,-1\n";
  gp << "plot '-' using 1:2 with lines title 'Fitness of best sample', '-' using 1:2 with lines title 'Average Fitness'\n";
  gp.send1d(plot_data);
  gp.send1d(plot_data_average);

  ///Weights of best sample once algorithm complete
  std::cout << "bishopPairWeight: " << '\n';
  std::cout << samples[selected_samples[0]][0] << '\n';

    std::cout << "shelterWeight: " << '\n';
  std::cout << samples[selected_samples[0]][1] << '\n';

    std::cout << "pawnStormWeight: " << '\n';
  std::cout << samples[selected_samples[0]][2] << '\n';

    std::cout << "kingEscapeWeight: " << '\n';
  std::cout << samples[selected_samples[0]][3] << '\n';

    std::cout << "kingTropismWeight: " << '\n';
  std::cout << samples[selected_samples[0]][4] << '\n';

    std::cout << "doubledPWeight: " << '\n';
  std::cout << samples[selected_samples[0]][5] << '\n';

    std::cout << "isolatedPWeight: " << '\n';
  std::cout << samples[selected_samples[0]][6] << '\n';

    std::cout << "PPWeight: " << '\n';
  std::cout << samples[selected_samples[0]][7] << '\n';

    std::cout << "rooksBehindWeight: " << '\n';
  std::cout << samples[selected_samples[0]][8] << '\n';

    std::cout << "backwardPWeight: " << '\n';
  std::cout << samples[selected_samples[0]][9] << '\n';

    std::cout << "chainsWeight: " << '\n';
  std::cout << samples[selected_samples[0]][10] << '\n';

    std::cout << "pawnMobWeight: " << '\n';
  std::cout << samples[selected_samples[0]][11] << '\n';

    std::cout << "phalanxWeight: " << '\n';
  std::cout << samples[selected_samples[0]][12] << '\n';

    std::cout << "outpostsWeight: " << '\n';
  std::cout << samples[selected_samples[0]][13] << '\n';

    std::cout << "knightMobWeight: " << '\n';
  std::cout << samples[selected_samples[0]][14] << '\n';

    std::cout << "bishopMobWeight: " << '\n';
  std::cout << samples[selected_samples[0]][15] << '\n';

    std::cout << "rookMobWeight: " << '\n';
  std::cout << samples[selected_samples[0]][16] << '\n';

    std::cout << "queenMobWeight: " << '\n';
  std::cout << samples[selected_samples[0]][17] << '\n';

    std::cout << "ccWeight: " << '\n';
  std::cout << samples[selected_samples[0]][18] << '\n';

    std::cout << "rookFileWeight: " << '\n';
  std::cout << samples[selected_samples[0]][19] << '\n';

    std::cout << "queenFileWeight: " << '\n';
  std::cout << samples[selected_samples[0]][20] << '\n';

    std::cout << "trapBishWeight: " << '\n';
  std::cout << samples[selected_samples[0]][21] << '\n';

    std::cout << "trapRookWeight: " << '\n';
  std::cout << samples[selected_samples[0]][22] << '\n';

    std::cout << "pinsWeight: " << '\n';
  std::cout << samples[selected_samples[0]][23] << '\n';

    std::cout << "forksWeight: " << '\n';
  std::cout << samples[selected_samples[0]][24] << '\n';

    std::cout << "skewersWeight: " << '\n';
  std::cout << samples[selected_samples[0]][25] << '\n';

    std::cout << "discAttWeight: " << '\n';
  std::cout << samples[selected_samples[0]][26] << '\n';

    std::cout << "doubAttWeight: " << '\n';
  std::cout << samples[selected_samples[0]][27] << '\n';

    std::cout << "overloadWeight: " << '\n';
  std::cout << samples[selected_samples[0]][28] << '\n';

    std::cout << "hangingWeight: " << '\n';
  std::cout << samples[selected_samples[0]][29] << '\n';

    std::cout << "egkingActWeight: " << '\n';
  std::cout << samples[selected_samples[0]][30] << '\n';

    std::cout << "rookCoordWeight: " << '\n';
  std::cout << samples[selected_samples[0]][31] << '\n';

    std::cout << "otherCoordWeight: " << '\n';
  std::cout << samples[selected_samples[0]][32] << '\n';

    std::cout << "kingCoordWeight: " << '\n';
  std::cout << samples[selected_samples[0]][33] << '\n';

    std::cout << "fiancettoWeight: " << '\n';
  std::cout << samples[selected_samples[0]][34] << '\n';

    std::cout << "developWeight: " << '\n';
  std::cout << samples[selected_samples[0]][35] << '\n';

  return 0;
}
