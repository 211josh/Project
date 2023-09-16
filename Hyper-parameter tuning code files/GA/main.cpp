///Genetic algorithm - tuning hyper-parameters via random search

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

const int no_features = 36; //no. features in evaluation function.

///FINDING BEST HYPER-PARAMETERS
const int no_runs = 250;

///HYPER-PARAMETERS
const int no_samples_min = 100, no_samples_max = 200;
const int no_positions = 500;
const int no_generations_min = 50, no_generations_max = 150;
const int select_percent_min = 5, select_percent_max = 20;
const float mutRate_min = 0.10, mutRate_max = 0.40;
const float mutRange_min = 1, mutRange_max = 10;
const int weight_range_min = 10, weight_range_max = 50;

///Best hyper-parameters
struct HP {
    int no_samples;
    int no_generations;
    int select_percent;
    float mutRate;
    float mutRange;
    int weight_range;
    float best_fitness;
};
HP bestHP; //best HP stored throughout.


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> distrib_samples(no_samples_min, no_samples_max);
std::uniform_int_distribution<> distrib_generations(no_generations_min, no_generations_max);
std::uniform_int_distribution<> distrib_select_percent(select_percent_min, select_percent_max);
std::uniform_real_distribution<> distrib_mutRate(mutRate_min, mutRate_max);
std::uniform_real_distribution<> distrib_mutRange(mutRange_min, mutRange_max);
std::uniform_int_distribution<> distrib_weight_range(weight_range_min, weight_range_max);

int eval_temp;

std::vector<int> current_weights(no_features); //weights used in evaluation function, updated for each sample.
std::vector<float> graph_eval_dif; //final eval dif of best sample after entire algorithm run
std::vector<int> graph_generation; //x-axis of graph, i.e no_runs.

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
  bestHP.best_fitness = 1000000000;

  Gnuplot gp;
  accessDB();
  for(int t = 0; t < no_runs; ++t){
      graph_generation.push_back(t+1); //add algorithm run to graph.
      std::cout << "ALGORITHM RUN: " << '\n';
      std::cout << t+1 << '\n';

      ///Random hyper-parameters within ranges
      int no_samples = distrib_samples(gen);
      int no_generations = distrib_generations(gen);
      int select_percent = distrib_select_percent(gen);
      float mutRate = distrib_mutRate(gen);
      float mutRange = distrib_mutRange(gen);
      int weight_range = distrib_weight_range(gen);

      const int no_saved = static_cast<int>(no_samples * (static_cast<float>(select_percent) / 100.0));
      int samples[no_samples][no_features]; //(no_samples) samples, each with one weight for each feature.
      int selected_samples[no_saved]; //samples which are selected.
      std::vector<int> eval_difference(no_samples); //fitness function for each sample, updated each generation.

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

      ///Main genetic algorithm loop
      for (int k = 0; k<no_generations; ++k){

            randomPositions.clear();

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
                    fen = fen4850[randomPositions[n]];
                    StateListPtr states(new std::deque<StateInfo>(1));
                    pos.set(fen, false, &states->back(), Threads.main());
                    ///Static-eval
                    Value v = Eval::evaluate(pos);
                    eval_temp = int(v)*100/PawnValueEg;
                    eval_difference[i] += fabs(eval_temp - evalDB4850[randomPositions[n]]); //overall difference in centipawns across positions.
                }
            }

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
              if (std::find(selected_samples, selected_samples + no_saved, i) != selected_samples + no_saved) {
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

      if(eval_difference[selected_samples[0]]/no_positions < bestHP.best_fitness){
            bestHP.no_samples = no_samples;
            bestHP.no_generations = no_generations;
            bestHP.select_percent = select_percent;
            bestHP.mutRate = mutRate;
            bestHP.mutRange = mutRange;
            bestHP.weight_range = weight_range;
            bestHP.best_fitness = (eval_difference[selected_samples[0]])/no_positions;

      }
      graph_eval_dif.push_back(bestHP.best_fitness); //eval_dif of best sample after algorithm
  }

  ///Graphing of performance
  gp << "set xlabel 'Algorithm run'\n";
  gp << "set ylabel 'Fitness of best sample found (centipawns)'\n";
  gp << "set title 'Smallest final eval difference achieved by hyper-parameters'\n";
  gp << "set yrange [0:*]\n";
  std::vector<std::pair<int,float>> plot_data;
  for(size_t i = 0; i < graph_generation.size(); ++i){
    plot_data.push_back(std::make_pair(graph_generation[i], graph_eval_dif[i]));
  }
  gp << "plot '-' with lines title 'Best Eval Diff'\n";
  gp.send1d(plot_data);

  std::cout << "BEST HYPER-PARAMETERS FOUND :" << '\n';

  std::cout << "no_samples: " << '\n';
  std::cout << bestHP.no_samples << '\n';

  std::cout << "no_positions: " << '\n';
  std::cout << no_positions << '\n';

  std::cout << "no_generations: " << '\n';
  std::cout << bestHP.no_generations << '\n';

  std::cout << "select_percent: " << '\n';
  std::cout << bestHP.select_percent << '\n';

  std::cout << "mutRate: " << '\n';
  std::cout << bestHP.mutRate << '\n';

  std::cout << "mutRange: " << '\n';
  std::cout << bestHP.mutRange << '\n';

  std::cout << "weight_range: " << '\n';
  std::cout << bestHP.weight_range << '\n';

  std::cout << "Fitness of best sample using hyper-parameters: " << '\n';
  std::cout << bestHP.best_fitness << '\n';

  return 0;
}
