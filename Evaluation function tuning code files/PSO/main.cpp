///PSO to tune the parameters of our evaluation function features.

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

///HYPER-PARAMETERS
const int no_particles = 100; //no. particles at each generation.
const int no_positions = 500; //no. positions in fitness evaluation. randomly selected each generation out of 4850 possible positions.
const int no_generations = 100; //max no. iterations of algorithm.
const float interia = 0.5; //how much of previous velocity retained. [0,1].
const float cognitive_c = 2;  //cognitive co-efficient - influence of particles own best position on its next. [1,3].
const float social_c = 2;  //social co-efficient - influence of best known particle position in swarm. [1,3].
const int weight_range = 30; //range for which weights can take value on interval [1,weight_range].
const float velocity_range = 4; //range of velocity. on interval [-velocity_range,velocity_range].

struct Particle {
    int position[no_features];  //position of each particle (weights)
    float velocity[no_features];  //velocity of particle
    int best_position[no_features];  //personal best position
    int best_score;  //best eval difference for the best position.
};

Particle particles[no_particles];
Particle global_best; //global best particle.

int eval_dif_average;

std::vector<int> current_weights(no_features); //weights used in evaluation function, updated for each particle.
std::vector<int> eval_difference(no_particles); //fitness function for each particle, updated each generation.
std::vector<float> graph_eval_dif; //used in graph. fitness function at each generation of global_best in centipawns.
std::vector<int> graph_generation; //x-axis of graph.
std::vector<float> graph_eval_dif_average;

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
  std::srand(std::time(nullptr)); //makes randomness completely random each run. remove for consistent randomness.
  Gnuplot gp;
  accessDB();

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
  std::cout << engine_info() << std::endl;
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
    std::cout << "GENERATION: " << '\n';
    std::cout << k+1 << '\n';
    randomPositions.clear();
    graph_generation.push_back(k+1); //Add generation number to graph.
    eval_dif_average = 0;

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
        eval_dif_average += eval_difference[i];

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
    graph_eval_dif.push_back(global_best.best_score/no_positions); //averaged out over number of positions in graph.
    graph_eval_dif_average.push_back((eval_dif_average/no_positions)/no_particles); //average sample fitness.


    ///Update particle positions and velocities
    std::uniform_real_distribution<> rdist(0.0, 1.0);
    for (int i = 0; i < no_particles; ++i) {
        for (int j = 0; j < no_features; ++j) {
            float r1 = rdist(gen);
            float r2 = rdist(gen);
            //update velocity:
            particles[i].velocity[j] = interia * particles[i].velocity[j] +
                                        cognitive_c * r1 * (particles[i].best_position[j] - particles[i].position[j]) +
                                        social_c * r2 * (global_best.best_position[j] - particles[i].position[j]);
            particles[i].velocity[j] = std::max(-velocity_range, std::min(particles[i].velocity[j], velocity_range));
            //update position:
            particles[i].position[j] += particles[i].velocity[j];
            particles[i].position[j] = std::max(1, std::min(particles[i].position[j], weight_range)); //keep in range [1,weight_range]
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
  gp << "set label ' " << global_best.best_score/no_positions << "' at " << graph_generation.back() << "," << global_best.best_score/no_positions << " point pointtype 7 offset char -5,-1\n";
  gp << "plot '-' using 1:2 with lines title 'Fitness of best sample', '-' using 1:2 with lines title 'Average Fitness'\n";
  gp.send1d(plot_data);
  gp.send1d(plot_data_average);

  ///Weights of global_best once algorithm complete
  std::cout << "bishopPairWeight: " << '\n';
  std::cout << global_best.best_position[0] << '\n';

    std::cout << "shelterWeight: " << '\n';
  std::cout << global_best.best_position[1] << '\n';

    std::cout << "pawnStormWeight: " << '\n';
  std::cout << global_best.best_position[2] << '\n';

    std::cout << "kingEscapeWeight: " << '\n';
  std::cout << global_best.best_position[3] << '\n';

    std::cout << "kingTropismWeight: " << '\n';
  std::cout << global_best.best_position[4] << '\n';

    std::cout << "doubledPWeight: " << '\n';
  std::cout << global_best.best_position[5] << '\n';

    std::cout << "isolatedPWeight: " << '\n';
  std::cout << global_best.best_position[6] << '\n';

    std::cout << "PPWeight: " << '\n';
  std::cout << global_best.best_position[7] << '\n';

    std::cout << "rooksBehindWeight: " << '\n';
  std::cout << global_best.best_position[8] << '\n';

    std::cout << "backwardPWeight: " << '\n';
  std::cout << global_best.best_position[9] << '\n';

    std::cout << "chainsWeight: " << '\n';
  std::cout << global_best.best_position[10] << '\n';

    std::cout << "pawnMobWeight: " << '\n';
  std::cout << global_best.best_position[11] << '\n';

    std::cout << "phalanxWeight: " << '\n';
  std::cout << global_best.best_position[12] << '\n';

    std::cout << "outpostsWeight: " << '\n';
  std::cout << global_best.best_position[13] << '\n';

    std::cout << "knightMobWeight: " << '\n';
  std::cout << global_best.best_position[14] << '\n';

    std::cout << "bishopMobWeight: " << '\n';
  std::cout << global_best.best_position[15] << '\n';

    std::cout << "rookMobWeight: " << '\n';
  std::cout << global_best.best_position[16] << '\n';

    std::cout << "queenMobWeight: " << '\n';
  std::cout << global_best.best_position[17] << '\n';

    std::cout << "ccWeight: " << '\n';
  std::cout << global_best.best_position[18] << '\n';

    std::cout << "rookFileWeight: " << '\n';
  std::cout << global_best.best_position[19] << '\n';

    std::cout << "queenFileWeight: " << '\n';
  std::cout << global_best.best_position[20] << '\n';

    std::cout << "trapBishWeight: " << '\n';
  std::cout << global_best.best_position[21] << '\n';

    std::cout << "trapRookWeight: " << '\n';
  std::cout << global_best.best_position[22] << '\n';

    std::cout << "pinsWeight: " << '\n';
  std::cout << global_best.best_position[23] << '\n';

    std::cout << "forksWeight: " << '\n';
  std::cout << global_best.best_position[24] << '\n';

    std::cout << "skewersWeight: " << '\n';
  std::cout << global_best.best_position[25] << '\n';

    std::cout << "discAttWeight: " << '\n';
  std::cout << global_best.best_position[26] << '\n';

    std::cout << "doubAttWeight: " << '\n';
  std::cout << global_best.best_position[27] << '\n';

    std::cout << "overloadWeight: " << '\n';
  std::cout << global_best.best_position[28] << '\n';

    std::cout << "hangingWeight: " << '\n';
  std::cout << global_best.best_position[29] << '\n';

    std::cout << "egkingActWeight: " << '\n';
  std::cout << global_best.best_position[30] << '\n';

    std::cout << "rookCoordWeight: " << '\n';
  std::cout << global_best.best_position[31] << '\n';

    std::cout << "otherCoordWeight: " << '\n';
  std::cout << global_best.best_position[32] << '\n';

    std::cout << "kingCoordWeight: " << '\n';
  std::cout << global_best.best_position[33] << '\n';

    std::cout << "fiancettoWeight: " << '\n';
  std::cout << global_best.best_position[34] << '\n';

    std::cout << "developWeight: " << '\n';
  std::cout << global_best.best_position[35] << '\n';

  return 0;
}
