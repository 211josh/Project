///PSO random-search to tune hyper-parameters

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

const int no_features = 36; //Number of features.

///FINDING BEST HYPER-PARAMETERS
const int no_runs = 250;

const int no_particles_min = 100, no_particles_max = 200;
const int no_positions = 500;
const int no_generations_min = 50, no_generations_max = 150;
const float inertia_min = 0.1, inertia_max = 0.9;
const float cognitive_c_min = 1, cognitive_c_max = 3;
const float social_c_min = 1, social_c_max = 3;
const int weight_range_min = 10, weight_range_max = 50;
const float velocity_range_min = 1, velocity_range_max = 4;

struct Particle {
    int position[no_features];  //position of each particle (weights)
    float velocity[no_features];  //velocity of particle
    int best_position[no_features];  //personal best position
    int best_score;  //best eval difference for the best position.
};

Particle global_best; //global best particle.

///Best hyper-parameters
struct HP {
    int no_particles;
    int no_generations;
    float inertia;
    float cognitive_c;
    float social_c;
    int weight_range;
    float velocity_range;
    float best_fitness;
};
HP bestHP; //best HP stored throughout.

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> distrib_particles(no_particles_min, no_particles_max);
std::uniform_int_distribution<> distrib_generations(no_generations_min, no_generations_max);
std::uniform_real_distribution<> distrib_inertia(inertia_min, inertia_max);
std::uniform_real_distribution<> distrib_cognitive(cognitive_c_min, cognitive_c_max);
std::uniform_real_distribution<> distrib_social(social_c_min, social_c_max);
std::uniform_int_distribution<> distrib_weight(weight_range_min, weight_range_max);
std::uniform_real_distribution<> distrib_vel(velocity_range_min,velocity_range_max);

std::vector<int> current_weights(no_features); //weights used in evaluation function, updated for each particle.
std::vector<float> graph_eval_dif; //used in graph. fitness function at each generation of global_best in centipawns.
std::vector<int> graph_generation; //x-axis of graph.

std::vector<std::string> fen4850(4850); //stores fens from database. [0 - 4849].
std::vector<int> evalDB4850(4850); //Stores static_evals from database. [0 - 4849].
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

  bestHP.best_fitness = 1000000;

  for(int t = 0; t < no_runs; ++t){

      graph_generation.push_back(t+1);
      std::cout << "ALGORITHM RUN: " << '\n';
      std::cout << t+1 << '\n';

      ///RANDOM HYPER-PARAMETERS WITHIN RANGE
      int no_particles = distrib_particles(gen);
      int no_generations = distrib_generations(gen);
      float inertia = distrib_inertia(gen);
      float cognitive_c = distrib_cognitive(gen);
      float social_c = distrib_social(gen);
      int weight_range = distrib_weight(gen);
      float velocity_range = distrib_vel(gen);

      std::vector<int> eval_difference(no_particles); //fitness function for each particle, updated each generation.
      Particle particles[no_particles];

      ///Initialize particles
      global_best.best_score = 100000000;
      for (int i = 0; i < no_particles; ++i) {
        for (int j = 0; j < no_features; ++j) {
            particles[i].position[j] = std::rand() % weight_range + 1;
            particles[i].velocity[j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (2 * velocity_range) - velocity_range;
        }
        particles[i].best_score = 100000000;
      }

      ///Initialisation required for evaluation function to be functioning.
      //std::cout << engine_info() << std::endl;
      UCI::init(Options);
      Tune::init();
      Bitboards::init();
      Position::init();
      Threads.set(size_t(Options["Threads"]));
      Search::clear();

      ///FEN initialisation
      std::string fen = "rn1qkbnr/ppp2ppp/8/3pp3/4P1b1/5N2/PPPP1PPP/RNB1KB1R w KQkq - 0 4"; //example fen.
      StateListPtr states(new std::deque<StateInfo>(1));
      Position pos;
      pos.set(fen, false, &states->back(), Threads.main());

      ///Main PSO loop
      for (int k = 0; k < no_generations; ++k){
        //std::cout << "GENERATION: " << '\n';
        //std::cout << k+1 << '\n';
        randomPositions.clear();

        ///RANDOM SELECTION OF POSITIONS FOR FITNESS FUNCTION
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 4849);
        for (int i = 0; i < no_positions; ++i) {
            int randomNumber = distrib(gen); //random number generated uniformly on interval [0,4849].
            randomPositions.push_back(randomNumber);
        }

        //for each particle
        for (int i = 0; i < no_particles; ++i) {
            eval_difference[i] = 0;
            for (int j = 0; j < no_features; ++j) {
                current_weights[j] = particles[i].position[j]; //set current_weights to particle weights for evaluation function.
            }
            ///Fen & Evaluations
            for(int n=0; n < no_positions; ++n){
                fen = fen4850[randomPositions[n]];
                StateListPtr states(new std::deque<StateInfo>(1));
                pos.set(fen, false, &states->back(), Threads.main());
                ///Static-eval
                Value v = Eval::evaluate(pos);
                float eval_temp = int(v)*100/PawnValueEg;
                eval_difference[i] += fabs(eval_temp - evalDB4850[randomPositions[n]]);
            }

            ///Update personal bests
            if (eval_difference[i] < particles[i].best_score) {
                particles[i].best_score = eval_difference[i];
                std::copy(std::begin(particles[i].position), std::end(particles[i].position), std::begin(particles[i].best_position));
            }

            ///Update global best
            if (eval_difference[i] < global_best.best_score) {
                global_best.best_score = eval_difference[i];
                std::copy(std::begin(particles[i].position), std::end(particles[i].position), std::begin(global_best.best_position));
            }
        }

        ///Update particle positions and velocities
        std::uniform_real_distribution<> rdist(0.0, 1.0);
        for (int i = 0; i < no_particles; ++i) {
            for (int j = 0; j < no_features; ++j) {
                float r1 = rdist(gen);
                float r2 = rdist(gen);
                //update velocity:
                particles[i].velocity[j] = inertia * particles[i].velocity[j] +
                                            cognitive_c * r1 * (particles[i].best_position[j] - particles[i].position[j]) +
                                            social_c * r2 * (global_best.best_position[j] - particles[i].position[j]);
                particles[i].velocity[j] = std::max(-velocity_range, std::min(particles[i].velocity[j], velocity_range));
                //update position:
                particles[i].position[j] += particles[i].velocity[j];
                particles[i].position[j] = std::max(1, std::min(particles[i].position[j], weight_range)); //keep in range [1,weight_range]
            }
        }
      }
    std::cout << "GLOBAL BEST: " << '\n';
    std::cout << global_best.best_score << '\n';
    std::cout << "BEST SO FAR: " << '\n';
    std::cout << bestHP.best_fitness << '\n';
    if(global_best.best_score < bestHP.best_fitness){
        bestHP.no_particles = no_particles;
        bestHP.no_generations = no_generations;
        bestHP.inertia = inertia;
        bestHP.cognitive_c = cognitive_c;
        bestHP.social_c = social_c;
        bestHP.weight_range = weight_range;
        bestHP.velocity_range = velocity_range;
        bestHP.best_fitness = global_best.best_score;
    }
    std::cout << (bestHP.best_fitness/no_positions) << '\n';
    graph_eval_dif.push_back(bestHP.best_fitness/no_positions);
    }


  ///Graphing of performance
  gp << "set xlabel 'Algorithm run'\n";
  gp << "set ylabel 'Fitness of best sample'\n";
  gp << "set title 'Best sample found within random-search'\n";
  gp << "set yrange [0:*]\n";
  std::vector<std::pair<int,float>> plot_data;
  for(size_t i = 0; i < graph_generation.size(); ++i){
    plot_data.push_back(std::make_pair(graph_generation[i], graph_eval_dif[i]));
  }
  gp << "plot '-' with lines title 'Best Eval Diff'\n";
  gp.send1d(plot_data);

  ///BEST HYPER-PARAMETERS FOUND:

  std::cout << "BEST HYPER-PARAMETERS FOUND: " << '\n';


  std::cout << "no_particles: " << '\n';
  std::cout << bestHP.no_particles << '\n';

  std::cout << "no_generations: " << '\n';
  std::cout << bestHP.no_generations << '\n';

  std::cout << "inertia: " << '\n';
  std::cout << bestHP.inertia << '\n';

  std::cout << "cognitive_c: " << '\n';
  std::cout << bestHP.cognitive_c << '\n';

  std::cout << "social_c: " << '\n';
  std::cout << bestHP.social_c << '\n';

  std::cout << "weight_range: " << '\n';
  std::cout << bestHP.weight_range << '\n';

  std::cout << "velocity_range: " << '\n';
  std::cout << bestHP.velocity_range << '\n';

  std::cout << "best_fitness: " << '\n';
  std::cout << (bestHP.best_fitness/no_positions) << '\n';



  return 0;
}

