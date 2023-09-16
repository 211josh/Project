///Evaluation function specifically for training.
///Used by all three methods: GA, SA & PSO.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <iostream>

#include "bitboard.h"
#include "evaluate.h"
#include "thread.h"
#include "types.h"

extern std::vector<int> current_weights;

namespace Stockfish {
namespace {

  //Evaluation class computes and stores attacks tables and other working data
  class Evaluation {
  public:
    Evaluation() = delete;
    explicit Evaluation(const Position& p) : pos(p) {}
    Evaluation& operator=(const Evaluation&) = delete;
    Value value();

  private:
    const Position& pos;
  };

  ///WEIGHTS - to be tuned

  int pawnWeight = 1, //FIXED
      knightWeight = 1, //FIXED
      bishopWeight = 1, //FIXED
      rookWeight = 1, //FIXED
      queenWeight = 1; //FIXED
//      bishopPairWeight = 1, //Rest are to be tuned between 1 and 10.
//      shelterWeight = 1,
//      pawnStormWeight = 1,
//      kingEscapeWeight = 1,
//      kingTropismWeight = 1,
//      doubledPWeight = 1,
//      isolatedPWeight = 1,
//      PPWeight = 1,
//      rooksBehindWeight = 1,
//      backwardPWeight = 1,
//      chainsWeight = 1,
//      pawnMobWeight = 1,
//      phalanxWeight = 1,
//      outpostsWeight = 1,
//      knightMobWeight = 1,
//      bishopMobWeight = 1,
//      rookMobWeight = 1,
//      queenMobWeight = 1,
//      ccWeight = 1,
//      rookFileWeight = 1,
//      queenFileWeight = 1,
//      trapBishWeight = 1,
//      trapRookWeight = 1,
//      pinsWeight = 1,
//      forksWeight = 1,
//      skewersWeight = 1,
//      discAttWeight = 1,
//      doubAttWeight = 1,
//      overloadWeight = 1,
//      hangingWeight = 1,
//      egkingActWeight = 1,
//      rookCoordWeight = 1,
//      otherCoordWeight = 1,
//      kingCoordWeight = 1,
//      fiancettoWeight = 1,
//      developWeight = 1;

  //Note - the output of the features does not necessarily matter, since the weights will determine the worth of the feature upon evaluation

  ///Material Score - general bonus for having pieces
  //Pawns
  int pawnScore(const Position& pos) {
  	return (pos.count<PAWN>(WHITE) - pos.count<PAWN>(BLACK))*100;
  }
  //Knights
  int knightScore(const Position& pos) {
  	return (pos.count<KNIGHT>(WHITE) - pos.count<KNIGHT>(BLACK))*300;
  }
  //Bishops
  int bishopScore(const Position& pos) {
  	return (pos.count<BISHOP>(WHITE) - pos.count<BISHOP>(BLACK))*300;
  }
  //Rooks
  int rookScore(const Position& pos) {
  	return (pos.count<ROOK>(WHITE) - pos.count<ROOK>(BLACK))*500;
  }
  //Queens
  int queenScore(const Position& pos) {
  	return (pos.count<QUEEN>(WHITE) - pos.count<QUEEN>(BLACK))*900;
  }
  //Bishop pair - bonus for having a pair of bishops.
  //bishops work together for many strategies so having them both is a bonus.
  int bishopPair(const Position& pos) {
  	bool whiteBishopPair = pos.count<BISHOP>(WHITE) == 2;
  	bool blackBishopPair = pos.count<BISHOP>(BLACK) == 2;
  	if (whiteBishopPair && !blackBishopPair) {
  		return 10;
  	}
  	else if (!whiteBishopPair && blackBishopPair) {
  		return -10;
  	}
  	return 0;
  }

  ///King Safety
  //King Shelter, i.e no. pawns in front of king.
  //checks for pawns on the rank infront of the king, the file to the left, right and the file of the king.
  int kingShelter(const Position& pos, Color c) {
    int score = 0;
    Square kingSquare = pos.square<KING>(c); //location of king
    int startFile = std::max(static_cast<int>(FILE_A), static_cast<int>(file_of(kingSquare)) - 1);
    int endFile = std::min(static_cast<int>(FILE_H), static_cast<int>(file_of(kingSquare)) + 1); //files around king
    Rank kingRank = rank_of(kingSquare);
    Rank frontRank = (c == WHITE) ? (kingRank < RANK_8 ? static_cast<Rank>(kingRank + 1) : RANK_8)
                                  : (kingRank > RANK_1 ? static_cast<Rank>(kingRank - 1) : RANK_1); //rank infront of king, depending on colour
    for (int f = startFile; f <= endFile; f++) { //loop through relevant files and ranks & check for pawns.
        Square s = make_square((File)f, frontRank);
        if (is_ok(s)) {
            Piece p = pos.piece_on(s);
            if (type_of(p) == PAWN && color_of(p) == c) {
                score += 3; //bonus of pawn.
            } else {
                score -= 3; //penalty for gap.
            }
        }
    }
    return score;
  }
  //Pawn storm, i.e incoming group of pawns
  //similar to king shelter, but for pawns of opposite colour, extra file on either side and extra rank infront.
  int pawnStorm(const Position& pos, Color c) {
    int score = 0;
    Square kingSquare = pos.square<KING>(c); //find king.
    Color enemy = ~c;
    int startFile = std::max(static_cast<int>(FILE_A), static_cast<int>(file_of(kingSquare)) - 2);
    int endFile = std::min(static_cast<int>(FILE_H), static_cast<int>(file_of(kingSquare)) + 2); //files of king, 2 each side.
    int frontRankInt = (c == WHITE) ? rank_of(kingSquare) + 1 : rank_of(kingSquare) - 1; //rank infront of king.
    int secondRankInt = (c == WHITE) ? rank_of(kingSquare) + 2 : rank_of(kingSquare) - 2; //2nd rank infront of king.
    if (frontRankInt < RANK_1) frontRankInt = RANK_1;
    if (frontRankInt > RANK_8) frontRankInt = RANK_8;
    if (secondRankInt < RANK_1) secondRankInt = RANK_1;
    if (secondRankInt > RANK_8) secondRankInt = RANK_8; //files & ranks bounded to the chess board.
    Rank frontRank = static_cast<Rank>(frontRankInt);
    Rank secondRank = static_cast<Rank>(secondRankInt);
    for (int f = startFile; f <= endFile; f++) { //loop through each file, checking ranks for enemy pawns.
      Square frontSquare = make_square((File)f, frontRank);
      Square secondSquare = make_square((File)f, secondRank);
      Piece frontPawn = pos.piece_on(frontSquare);
      Piece secondPawn = pos.piece_on(secondSquare);
      if (frontPawn == PAWN && color_of(frontPawn) == enemy) {
        score -= 4;  //penalty for enemy pawn 1 rank ahead.
      }
      else if (secondPawn == PAWN && color_of(secondPawn) == enemy) {
        score -= 2;  //enemy pawn 2 ranks ahead, smaller threat, less penalty.
      }
    }
    return score;
  }
  //King escape, i.e mobility of king to safe squares.
  //if a square around a king is attacked, it decreases the safety of the king.
  int kingEscapeRoutes(const Position& pos, Color c) {
    int score = 0;
    Square kingSquare = pos.square<KING>(c);
    Color enemy = ~c;
    int dx[] = {-1, 0, 1, 1, 1, 0, -1, -1};
    int dy[] = {1, 1, 1, 0, -1, -1, -1, 0}; //possible moves of king
    for (int i = 0; i < 8; i++) { //check each square around king
        Rank rank = static_cast<Rank>(rank_of(kingSquare) + dy[i]);
        File file = static_cast<File>(file_of(kingSquare) + dx[i]);
        if (file >= FILE_A && file <= FILE_H && rank >= RANK_1 && rank <= RANK_8) {
            Square dest = make_square(file, rank);
            if (!pos.attackers_to(dest, enemy)) { //check if square attacked
                score += 4; //bonus for safe square.
            } else {
                score -= 4; //penalty if square is attacked.
            }
        }
    }
    return score;
  }
  //King tropism, i.e distance of enemy major pieces to king
  //considers distance of king and attacking pieces. more opponent pieces around king decrease safety.
  int kingTropism(const Position& pos, Color c) {
    int score = 0;
    Square kingSquare = pos.square<KING>(c);
    Color enemy = ~c;
    Bitboard enemyQueens = pos.pieces(enemy, QUEEN);
    Bitboard enemyRooks = pos.pieces(enemy, ROOK);
    Bitboard enemyBishops = pos.pieces(enemy, BISHOP);
    auto distance_sq = [&](Square a, Square b) {
        int dx = file_of(a) - file_of(b);
        int dy = rank_of(a) - rank_of(b);
        return std::max(std::abs(dx), std::abs(dy)); //distance calc between two squares.
    };
    while (enemyQueens) { //for each enemy queen:
        Square s = pop_lsb(&enemyQueens);
        int distance = distance_sq(kingSquare, s);
        score -= 20 / (float)distance;  //penalty decreases as distance increases.
    }
    while (enemyRooks) { //for each enemy rook:
        Square s = pop_lsb(&enemyRooks);
        int distance = distance_sq(kingSquare, s);
        score -= 12 / (float)distance;
    }
    while (enemyBishops) { //for each enemy bishop:
        Square s = pop_lsb(&enemyBishops);
        int distance = distance_sq(kingSquare, s);
        score -= 6 / (float)distance;
    }
    return score;
}

  ///Pawn Structure
  //Doubled Pawns, i.e two pawns on same file.
  //when two pawns are on the same file, it restricts their mobility and increases their vulnerability.
  int doubledPawns(const Position& pos, Color c) {
    int score = 0;
    for (File f = FILE_A; f <= FILE_H; ++f) { //loop through files:
        if (popcount(pos.pieces(c, PAWN) & file_bb(f)) > 1) { //if no. pawns in file > 1:
            score -= 4;  //penalty of for each set of doubled pawns.
        }
    }
    return score;
  }
  //Isolated Pawns, ie no pawns on adjacent files of a pawn.
  //if a pawn is isolated, it cannot be defended by another pawn, making it vulnerable.
  int isolatedPawns(const Position& pos, Color c) {
    int score = 0;
    Bitboard myPawns = pos.pieces(c, PAWN);
    for (File f = FILE_A; f <= FILE_H; ++f) { //loop through files:
        if (f != FILE_A && f != FILE_H) {
            if ((myPawns & file_bb(f)) && !(myPawns & (file_bb(File(f - 1)) | file_bb(File(f + 1))))) { //if pawn is on file but not on adjacent:
                score -= 6;  //penalty for each isolated pawn.
            }
        }
    }
    return score;
  }
  //Passed Pawns
  //a passed pawn is a pawn which has no pawn opposition to promote, on its file and adjacent files.
  int passedPawns(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    while (pawns) { //for each pawn:
        Square sq = pop_lsb(&pawns);
        Bitboard mask = 0ULL; //pawns path to promotion.
        int rank = rank_of(sq);
        int file = file_of(sq);
        if (c == WHITE) { //for white pawn:
            for (int r = rank + 1; r <= 7; ++r) {
                int target_square = r * 8 + file;
                mask |= 1ULL << target_square;
                if (file > 0) mask |= 1ULL << (target_square - 1);
                if (file < 7) mask |= 1ULL << (target_square + 1); //including adjacent files
            }
        } else { //for black pawn:
            for (int r = rank - 1; r >= 0; --r) {
                int target_square = r * 8 + file;
                mask |= 1ULL << target_square;
                if (file > 0) mask |= 1ULL << (target_square - 1);
                if (file < 7) mask |= 1ULL << (target_square + 1);
            }
        }
        if (!(pos.pieces(~c, PAWN) & mask)) { //if no pawn opposition in mask:
            score += 4; //bonus for passed pawn.
        }
    }
    return score;
  }
  //Rooks behind passed pawn:
  //if a rook is behind a passed pawn, it adds a defender for it's entire path to promotion.
  int rooksBehindPP(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    Bitboard rooks = pos.pieces(c, ROOK);
    while (pawns) {
        Square pawn_sq = pop_lsb(&pawns);
        int pawn_rank = rank_of(pawn_sq);
        int pawn_file = file_of(pawn_sq);
        Bitboard mask = 0ULL;
        if (c == WHITE) {
            for (int r = pawn_rank + 1; r <= 7; ++r) {
                int target_square = r * 8 + pawn_file;
                mask |= 1ULL << target_square;
            }
        } else {
            for (int r = pawn_rank - 1; r >= 0; --r) {
                int target_square = r * 8 + pawn_file;
                mask |= 1ULL << target_square;
            }
        }
        if (pos.pieces(~c, PAWN) & mask) {
            continue; //same code as previous feature. given we have a passed pawn, check for rook behind:
        }
        int rook_square = (c == WHITE ? pawn_rank - 1 : pawn_rank + 1) * 8 + pawn_file;
        if (rooks & (1ULL << rook_square)) {
            score += 3; //bonus if rook behind passed pawn
        }
    }
    return score;
  }
  //Backward Pawns, i.e a pawn which has no adjacent pawns for support and will be captured if it advances:
  //these pawns are vulnerable to attack, so penalty.
  int backwardPawns(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    while (pawns) { //for each pawn:
        Square pawn_sq = pop_lsb(&pawns);
        int pawn_rank = rank_of(pawn_sq);
        int pawn_file = file_of(pawn_sq);
        bool has_adjacent_support = false;
        if (pawn_file > 0 && pos.pieces(c, PAWN) & (1ULL << (pawn_rank * 8 + pawn_file - 1))) {
            has_adjacent_support = true;
        }
        if (pawn_file < 7 && pos.pieces(c, PAWN) & (1ULL << (pawn_rank * 8 + pawn_file + 1))) {
            has_adjacent_support = true;
        }
        if (has_adjacent_support) {
            continue; //skip pawn if it has adjacent support.
        }
        int forward_square = (c == WHITE ? pawn_rank + 1 : pawn_rank - 1) * 8 + pawn_file;
        Bitboard forward_mask = 1ULL << forward_square;
        if (pos.pieces(~c, PAWN) & forward_mask) {
            continue; //skip pawn if pawn opposition directly in front.
        }
        if ((pawn_file > 0 && pos.pieces(~c, PAWN) & (1ULL << (forward_square - 1))) ||
            (pawn_file < 7 && pos.pieces(~c, PAWN) & (1ULL << (forward_square + 1)))) { //and if pawn opposition can capture upon advancement:
            score -= 3; //penalty for backward pawn.
        }
    }
    return score;
  }
  //Pawn Chains
  //a pawn chain is a chain of pawns on a long diagonal with only one undefended.
  //we simplify this by giving bonuses when pawns defend other pawns.
  int pawnChains(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    while (pawns) { //for each pawn:
        Square pawn_sq = pop_lsb(&pawns);
        int pawn_rank = rank_of(pawn_sq);
        int pawn_file = file_of(pawn_sq);
        int forward_left = (c == WHITE ? pawn_rank + 1 : pawn_rank - 1) * 8 + pawn_file - 1; //pawn left&right diagonals.
        int forward_right = (c == WHITE ? pawn_rank + 1 : pawn_rank - 1) * 8 + pawn_file + 1;
        if (pawn_file > 0 && (pos.pieces(c, PAWN) & (1ULL << forward_left))) { //if friendly pawn on left diagonal, i.e defending it:
            score += 4;
        }
        if (pawn_file < 7 && (pos.pieces(c, PAWN) & (1ULL << forward_right))) { //if friendly pawn on right diagonal:
            score += 4; //bonus for pawn chain
        }
    }
    return score;
}
  //Pawn Mobility
  //the availability for a pawn to push forward, i.e no pawn opposition directly ahead.
  int pawnMobility(const Position& pos, Color c) {
    int score = 0;
    Bitboard myPawns = pos.pieces(c, PAWN);
    Bitboard enemyPawns = pos.pieces(~c, PAWN);

    while (myPawns) { //for each pawn:
        Square s = pop_lsb(&myPawns);
        Square ahead = s + pawn_push(c);
        if (!(ahead & enemyPawns)) { //if pawn ahead has no pawn opposition:
            score += 1; //bonus for possibility of pawn advancement.
        }
    }
    return score;
  }
  //Pawn Phalanx
  //similar to pawn chains, but when pawns are on adjacent files & same rank.
  //seen as bonus because they control a wall of squares ahead of them, plus pushing a pawn guarantees adjacent pawn support.
  int pawnPhalanx(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    while (pawns) { //for each pawn:
        Square pawn_sq = pop_lsb(&pawns);
        int pawn_rank = rank_of(pawn_sq);
        int pawn_file = file_of(pawn_sq);
        if (pawn_file > 0 && (pos.pieces(c, PAWN) & (1ULL << (pawn_rank * 8 + pawn_file - 1)))) {
            score += 4; //bonus for pawn on left.
        }
        if (pawn_file < 7 && (pos.pieces(c, PAWN) & (1ULL << (pawn_rank * 8 + pawn_file + 1)))) {
            score += 4; //bonus for pawn on right.
        }
    }
    return score;
  }
  //Outposts
  //pawn outposts are squares on the opponents 4 ranks which is protected by a friendly pawn which cannot be attacked by pawn opposition.
  //pawn outposts stand as an anchor point for pieces, increasing their range of attack while granting safety.
  int pawnOutposts(const Position &pos, Color c) {
    int score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    while (pawns) { //for each pawn:
        Square pawn_sq = pop_lsb(&pawns);
        int pawn_rank = rank_of(pawn_sq);
        int pawn_file = file_of(pawn_sq);
        if ((c == WHITE && pawn_rank < 4) || (c == BLACK && pawn_rank > 3)) {
            continue; //skip pawns in friendly 4 ranks.
        }
        int left_attack = (c == WHITE ? pawn_rank - 1 : pawn_rank + 1) * 8 + pawn_file - 1;
        int right_attack = (c == WHITE ? pawn_rank - 1 : pawn_rank + 1) * 8 + pawn_file + 1;
        if ((pawn_file > 0 && (pos.pieces(~c, PAWN) & (1ULL << left_attack))) ||
            (pawn_file < 7 && (pos.pieces(~c, PAWN) & (1ULL << right_attack)))) {
            continue; //skip pawns which can be attacked.
        }
        score += 6; //bonus for pawn outpost.
    }
    return score;
  }

  ///Piece Mobility
  //Piece mobility is an advantage as it increases your options of play, pieces cover more squares etc.
  //Knights
  int knightMobility(const Position &pos, Color c) {
    int score = 0;
    Bitboard knights = pos.pieces(c, KNIGHT);
    Bitboard opponent_pieces = pos.pieces(~c);
    Bitboard empty_or_opponent_pieces = ~(pos.pieces(c)) | opponent_pieces;
    static const int KnightMoves[8][2] = { //all possible knight moves.
        {2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}
    };
    while (knights) { //for each knight:
        Square knight_sq = pop_lsb(&knights);
        int knight_rank = rank_of(knight_sq);
        int knight_file = file_of(knight_sq); //locate knight
        for (int i = 0; i < 8; i++) { //for each knight move:
            int new_rank = knight_rank + KnightMoves[i][0];
            int new_file = knight_file + KnightMoves[i][1];
            if (new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8) { //ensure move is within board.
                Square new_sq = Square(new_rank * 8 + new_file);
                if (empty_or_opponent_pieces & (1ULL << new_sq)) { //if the move goes on an empty square or captures a piece:
                    score += 3; //bonus for each knight move
                }
            }
        }
    }
    return score;
  }
  //Bishops
  int bishopMobility(const Position &pos, Color c) {
    int score = 0;
    Bitboard bishops = pos.pieces(c, BISHOP);
    Bitboard empty_or_opponent_pieces = ~(pos.pieces(c));
    static const int DiagonalMoves[4][2] = { //all possible directions of movement of bishop.
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    while (bishops) { //for each bishop:
        Square bishop_sq = pop_lsb(&bishops);
        int bishop_rank = rank_of(bishop_sq);
        int bishop_file = file_of(bishop_sq); //locate bishop.
        for (int i = 0; i < 4; i++) { //for each move direction:
            int new_rank = bishop_rank;
            int new_file = bishop_file;
            while (true) { //while all follow criteria met:
                new_rank += DiagonalMoves[i][0];
                new_file += DiagonalMoves[i][1];

                if (new_rank < 0 || new_rank >= 8 || new_file < 0 || new_file >= 8) { //break if move goes off board.
                    break;
                }
                Square new_sq = Square(new_rank * 8 + new_file);
                if (pos.pieces(c) & (1ULL << new_sq)) { //break if blocked by friendly piece.
                    break;
                }
                score += 2; //bonus for each bishop move.
                if (pos.pieces(~c) & (1ULL << new_sq)) { //break if blocked by enemy piece (done after since capturing is a move).
                    break;
                }
            }
        }
    }
    return score;
  }
  //Rooks
  int rookMobility(const Position &pos, Color c) {
    int score = 0;
    Bitboard rooks = pos.pieces(c, ROOK);
    Bitboard empty_or_opponent_pieces = ~(pos.pieces(c));
    static const int StraightMoves[4][2] = { //all directions of movement.
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}
    };
    while (rooks) { //for each rook:
        Square rook_sq = pop_lsb(&rooks);
        int rook_rank = rank_of(rook_sq);
        int rook_file = file_of(rook_sq);
        for (int i = 0; i < 4; i++) {
            int new_rank = rook_rank;
            int new_file = rook_file;
            while (true) { //while all criteria met:
                new_rank += StraightMoves[i][0];
                new_file += StraightMoves[i][1];
                if (new_rank < 0 || new_rank >= 8 || new_file < 0 || new_file >= 8) { //break is move goes off board.
                    break;
                }
                Square new_sq = Square(new_rank * 8 + new_file);
                if (pos.pieces(c) & (1ULL << new_sq)) { //break if blocked by friendly piece.
                    break;
                }
                score += 2; //bonus for each rook move
                if (pos.pieces(~c) & (1ULL << new_sq)) { //break if blocked by enemy piece.
                    break;
                }
            }
        }
    }
    return score;
  }
  //Queens
  //Same method as bishop & rooks but combined.
  int queenMobility(const Position &pos, Color c) {
    int score = 0;
    Bitboard queens = pos.pieces(c, QUEEN);
    static const int QueenMoves[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    while (queens) {
        Square queen_sq = pop_lsb(&queens);
        int queen_rank = rank_of(queen_sq);
        int queen_file = file_of(queen_sq);
        for (int i = 0; i < 8; i++) {
            int new_rank = queen_rank;
            int new_file = queen_file;
            while (true) {
                new_rank += QueenMoves[i][0];
                new_file += QueenMoves[i][1];
                if (new_rank < 0 || new_rank >= 8 || new_file < 0 || new_file >= 8) {
                    break;
                }
                Square new_sq = Square(new_rank * 8 + new_file);
                if (pos.pieces(c) & (1ULL << new_sq)) {
                    break;
                }
                score += 1; //bonus for each queen move.
                if (pos.pieces(~c) & (1ULL << new_sq)) {
                    break;
                }
            }
        }
    }
    return score;
  }
  //Central Control
  //central control is important, as the centre of the board allows for mobility and easy access to rest of board.
  int centralControl(const Position &pos, Color c) {
    int score = 0;
    Bitboard central_squares = (1ULL << Square(27)) | (1ULL << Square(28))
                            | (1ULL << Square(35)) | (1ULL << Square(36)); //central 4 squares.
    Bitboard central_pawns = pos.pieces(c, PAWN) & central_squares;
    score += popcount(central_pawns) * 2; //if occupied by friendly pawn, bonus.
    Bitboard central_pieces = pos.pieces(c) & ~pos.pieces(c, PAWN) & central_squares;
    score += popcount(central_pieces) * 1; //if occupied by friendly major piece, bonus.

    return score;
  }
  //Open & Semi-Open Files (rooks & queens)
  //rooks & queens on open/semi-open files allow for more control of the board.
  int rookOnFile(const Position& pos, Color c) {
    int score = 0;
    Bitboard rooks = pos.pieces(c, ROOK);
    while (rooks) { //for each rook:
        Square s = pop_lsb(&rooks);
        File f = file_of(s); //get file of rook.
        if (!pos.pieces(PAWN) & file_bb(f)) { //if fully open:
            score += 4;  //bonus
        } else if (!(pos.pieces(c, PAWN) & file_bb(f))) { //if semi-open (only pawn opposition)
            score += 2;  //bonus
        }
    }
    return score;
  }
  int queenOnFile(const Position& pos, Color c) {
    int score = 0;
    Bitboard queens = pos.pieces(c, QUEEN);
    while (queens) { //for each queen:
        Square s = pop_lsb(&queens);
        File f = file_of(s); //get file of queen.
        if (!pos.pieces(PAWN) & file_bb(f)) { //if open:
            score += 3;  //bonus.slightly less as queens are easily threatened by rook opposition.
        } else if (!(pos.pieces(c, PAWN) & file_bb(f))) {
            score += 2;  //bonus
        }
    }
    return score;
  }

  ///Trapped Pieces
  //Bishop trapped by pawns on a7/h7 or a2/h2 - hard-coded since it's very common
  int trappedBishop(const Position& pos) {
    int score = 0;
    if (pos.piece_on(SQ_A7) == BISHOP && color_of(pos.piece_on(SQ_A7)) == BLACK && pos.piece_on(SQ_B6) == PAWN && color_of(pos.piece_on(SQ_B6)) == WHITE)
      score += 20;
    if (pos.piece_on(SQ_H7) == BISHOP && color_of(pos.piece_on(SQ_H7)) == BLACK && pos.piece_on(SQ_G6) == PAWN && color_of(pos.piece_on(SQ_G6)) == WHITE)
        score += 20;
    if (pos.piece_on(SQ_A2) == BISHOP && color_of(pos.piece_on(SQ_A2)) == WHITE && pos.piece_on(SQ_B3) == PAWN && color_of(pos.piece_on(SQ_B3)) == BLACK)
        score -= 20;
    if (pos.piece_on(SQ_H2) == BISHOP && color_of(pos.piece_on(SQ_H2)) == WHITE && pos.piece_on(SQ_G3) == PAWN && color_of(pos.piece_on(SQ_G3)) == BLACK)
        score -= 20;
    return score;
  }
  //Rook stuck because of king
  //kings can block rooks from moving an entire direction on the back rank if king is blocking it
  int rookTrappedByKing(const Position& pos, Color c) {
  int score = 0;
  Square ksq = pos.square<KING>(c);
  if (rank_of(ksq) == (c == WHITE ? RANK_1 : RANK_8)) { //if king on back rank:
      if (file_of(ksq) < FILE_E && pos.piece_on(make_square(static_cast<File>(file_of(ksq) + 1), rank_of(ksq))) == ROOK)
          score -= 5; //if king on left side of board with rook stuck on right.
      if (file_of(ksq) > FILE_E && pos.piece_on(make_square(static_cast<File>(file_of(ksq) - 1), rank_of(ksq))) == ROOK)
          score -= 5; //vice versa.
  }
  return score;
  }

  ///Tactics
  ///Some tactics are simplified for the sake of complexity
  //Pins
  //a pin is when a piece cannot be moved since it exposes an attack on the king.
  //this feature represents the current pins and the possibility of pins.
  int havePins(const Position& pos, Color c) {
    int score = 0;
    Square ksq = pos.square<KING>(c);
    Direction directions[] = {NORTH, SOUTH, EAST, WEST, NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST};
    for (Direction d : directions) {
      Square sq = ksq;
      while (true) { //for each direction from king square:
        sq += d;
        if (sq < SQ_A1 || sq > SQ_H8) break; //check in board.
        Piece piece = pos.piece_on(sq);
        Color pieceColor = color_of(piece);
        if (pieceColor != ~c) break;
        if (pieceColor == c) { //if friendly piece on direction from king:
          PieceType type = type_of(piece);
          if (type == PAWN || type == KNIGHT || type == BISHOP || type == ROOK || type == QUEEN) {
            if (type == PAWN) {
              score -= 10; //penalty if friendly pawn pinned/possibility of pin.
            } else {
              score -= 20; //penalty if friendly major piece pinned/possibility.
            }
          }
          break;
        }
      }
    }

    return score;
  }
  //Knight Forks
  //a knight fork is when a knight is attacking multiple pieces at once, a tactic which usually results in material gain.
  int haveForks(const Position& pos, Color c) {
    int score = 0;
    Bitboard knights = pos.pieces(c, KNIGHT);
    while (knights) { //for each knight:
      Square s = pop_lsb(&knights);
      Bitboard attacks = pos.attacks_from<KNIGHT>(s) & pos.pieces(~c); //check opponent pieces the knight attacks.
      if (popcount(attacks) > 1) { //if it's more than 1, we have a fork:
          while (attacks) { //for each attack:
              Square target = pop_lsb(&attacks);
              PieceType attackedPiece = type_of(pos.piece_on(target));
              switch (attackedPiece) {
                  case PAWN: score += 7; break; //check which piece is attacked and score accordingly.
                  case KNIGHT: score += -4; break; //Forking a knight means it can take your knight => usually not good.
                  case BISHOP: score += 11; break;
                  case ROOK: score += 15; break;
                  case QUEEN: score += 23; break;
                  case KING: score += 27; break;
                  default: break;
              }
          }
      }
    }
    return score;
  }
  //Skewers
  //when attacking a valuable piece forces it to move, exposing a piece behind it.
  //king not included as valuable piece as it is technically included in pins.
  int haveSkewers(const Position& pos, Color c) {
    int score = 0;
    Color enemy = ~c;
    Bitboard skewerPieces = pos.pieces(c, BISHOP) | pos.pieces(c, ROOK) | pos.pieces(c, QUEEN); //possible skewering pieces.
    while (skewerPieces) { //for each:
        Square s = pop_lsb(&skewerPieces);
        PieceType skewerPiece = type_of(pos.piece_on(s));
        Bitboard attacks;
        switch (skewerPiece) { //identify possible attacks given piece type:
            case BISHOP:
                attacks = pos.attacks_from<BISHOP>(s);
                break;
            case ROOK:
                attacks = pos.attacks_from<ROOK>(s);
                break;
            case QUEEN:
                attacks = pos.attacks_from<QUEEN>(s);
                break;
            default:
                continue;
        }
        Bitboard valuableTargets = attacks & (pos.pieces(enemy, QUEEN) | pos.pieces(enemy, ROOK)); //valuable piece & being attacked:
        while (valuableTargets) { //for each valuable piece:
            Square target = pop_lsb(&valuableTargets);
            Bitboard behindTargets = between_bb(s, target) & (pos.pieces(enemy) ^ (pos.pieces(enemy, KING) | pos.pieces(enemy, QUEEN) | pos.pieces(enemy, ROOK)));
            if (behindTargets) { //if there is a piece behind it:
                score += 10;  //bonus for skewered piece (from perspective of skewering the enemy)
            }
        }
    }
    return score;
  }
  //Discovered Attacks
  //when moving a friendly piece exposes an attack on an enemy from a different friendly piece
  //feature explores potential discovered attacks, not certain discovered attacks, for computational simplicity.
  int discoveredAttacks(const Position& pos, Color c) {
    int score = 0;
    Bitboard bishops = pos.pieces(c, BISHOP);
    Bitboard rooks = pos.pieces(c, ROOK);
    Bitboard queens = pos.pieces(c, QUEEN);
    Bitboard sliders = bishops | rooks | queens; //sliding pieces which can perform discovered attacks.
    while (sliders) { //for each sliding piece:
        Square s = pop_lsb(&sliders);
        Bitboard possibleDiscoveries = pos.attacks_from<QUEEN>(s) & pos.pieces(c); //pieces which could move to reveal discovered attack.
        //queen for simplicity
        while (possibleDiscoveries) { //for each of these:
            Square uncoveringSquare = pop_lsb(&possibleDiscoveries);
            Piece uncoveringPiece = pos.piece_on(s);
            Bitboard targets = 0;
            switch (uncoveringPiece) { //what pieces can the moving piece attack:
                case PAWN:
                    targets = pos.attacks_from<PAWN>(uncoveringSquare, c);
                    break;
                case KNIGHT:
                    targets = pos.attacks_from<KNIGHT>(uncoveringSquare);
                    break;
                case BISHOP:
                    targets = pos.attacks_from<BISHOP>(uncoveringSquare);
                    break;
                case ROOK:
                    targets = pos.attacks_from<ROOK>(uncoveringSquare);
                    break;
                case KING:
                    targets = pos.attacks_from<KING>(uncoveringSquare);
                    break;
                default:
                    continue;
            }
            targets &= pos.pieces(~c); //if the piece is an enemy piece
            if (targets) {
                score += 10; //bonus for potential discovered attack
            }
        }
    }

    return score;
  }
  //Double Attacks
  //when a friendly piece is attacking more than one enemy piece at once
  int doubleAttacks(const Position& pos, Color c) {
    int score = 0;
    Color enemy = ~c;
    Bitboard currentPieces = pos.pieces(c);
    while (currentPieces) { //for each friendly piece:
        Square s = pop_lsb(&currentPieces);
        PieceType pt = type_of(pos.piece_on(s)); //identify piece type.
        Bitboard attackedSquares;
        switch (pt) { //find all squares attacked by this piece.
            case PAWN:
                attackedSquares = pos.attacks_from<PAWN>(s) & pos.pieces(~c);
                break;
            case KNIGHT:
                attackedSquares = pos.attacks_from<KNIGHT>(s);
                break;
            case BISHOP:
                attackedSquares = pos.attacks_from<BISHOP>(s);
                break;
            case ROOK:
                attackedSquares = pos.attacks_from<ROOK>(s);
                break;
            case QUEEN:
                attackedSquares = pos.attacks_from<QUEEN>(s);
                break;
            case KING:
                attackedSquares = pos.attacks_from<KING>(s);
                break;
            default:
                assert(false);
                break;
        }
        Bitboard attackedEnemies = attackedSquares & pos.pieces(enemy);
        if (popcount(attackedEnemies) > 1) { //if no. attacked enemy pieces > 1: double attack:
            switch (pt) {
                case PAWN:
                    score += 5; //score based on which piece is performing double attack
                    break;
                case KNIGHT:
                    score += 10;
                    break;
                case BISHOP:
                    score += 10;
                    break;
                case ROOK:
                    score += 14;
                    break;
                case QUEEN:
                    score += 18;
                    break;
                case KING:
                    score += 5;
                    break;
                default:
                    assert(false);
                    break;
            }
        }
    }
    return score;
  }
  //Overloaded pieces
  //when a piece is defending a lot of pieces it becomes overloaded.
  //we slightly simplify this concept by applying a penalty if a piece is being attacked more than it is being defended.
  bool isOverloaded(const Position& pos, Square s, Color defender) {
    Bitboard attackers = pos.attackers_to(s);
    Bitboard defenders = pos.attackers_to(s, pos.pieces(defender)) & pos.pieces(defender);
    return popcount(attackers) >= popcount(defenders);
  }
  int haveOverloads(const Position& pos, Color c) {
    int score = 0;
    Color enemy = ~c;
    Bitboard friendlyPieces = pos.pieces(c);
    while (friendlyPieces) { //for each friendly piece:
        Square s = pop_lsb(&friendlyPieces);
        PieceType pt = type_of(pos.piece_on(s));
        if (isOverloaded(pos, s, c)) { //if piece is overloaded
            switch (pt) {
                case PAWN:
                    score -= 3; //penalty based on piece
                    break;
                case KNIGHT:
                    score -= 6;
                    break;
                case BISHOP:
                    score -= 6;
                    break;
                case ROOK:
                    score -= 8;
                    break;
                case QUEEN:
                    score -= 12;
                    break;
                case KING:
                    score -= 14;
                    break;
                default:
                    assert(false);
                    break;
            }
        }
    }
    return score;
  }
  //Hanging Pieces
  //a piece which is being attacked but not defended, which makes it more vulnerable.
  bool is_attacked(const Position& pos, Square s, Color attacker) {
    return pos.attackers_to(s) & pos.pieces(attacker); //check if piece is attacked.
  }
  bool is_defended(const Position& pos, Square s, Color defender) {
    return pos.attackers_to(s, pos.pieces(defender)) & pos.pieces(defender); //check if piece is defended.
  }
  int hangingPieces(const Position& pos, Color c) {
    int score = 0;
    Color enemy = ~c;
    Bitboard friendlyPieces = pos.pieces(c);
    while (friendlyPieces) { //for each friendly piece:
        Square s = pop_lsb(&friendlyPieces);
        PieceType pt = type_of(pos.piece_on(s));
        if (is_attacked(pos, s, enemy) && !is_defended(pos, s, c)) { //piece is hanging:
            switch (pt) {
                case PAWN:
                    score -= 3; //assign penalty based on piece.
                    break;
                case KNIGHT:
                    score -= 7;
                    break;
                case BISHOP:
                    score -= 7;
                    break;
                case ROOK:
                    score -= 11;
                    break;
                case QUEEN:
                    score -= 19;
                    break;
                case KING:
                    break;
                default:
                    assert(false);
                    break;
            }
        }
    }
    return score;
  }

  ///Endgame Features
  //Check if the game is in endgame
  bool isEndgame(const Position& pos) {
  	const int endgameThreshold = 20; //"isEndgame" determined by amount of material on the board.
  	const int boardMaterial = (pos.count<PAWN>(WHITE) + pos.count<PAWN>(BLACK))
  		+ (pos.count<KNIGHT>(WHITE) + pos.count<KNIGHT>(BLACK))*3
  		+ (pos.count<BISHOP>(WHITE) + pos.count<BISHOP>(BLACK))*3
  		+ (pos.count<ROOK>(WHITE) + pos.count<ROOK>(BLACK))*5
  		+ (pos.count<QUEEN>(WHITE) + pos.count<QUEEN>(BLACK))*9;
  	return(boardMaterial<endgameThreshold);
  }
  //King Activity
  //in endgame, the king needs to become more active on the board. it's necessary to check if it's endgame because we don't want an active king
  //in the opening or middle game.
  int egkingActivity(const Position& pos, Color c) {
    Square kingSq = pos.square<KING>(c);
    int distanceToCentre = std::min({ distance(kingSq, SQ_D4), distance(kingSq, SQ_E4),
                                      distance(kingSq, SQ_D5), distance(kingSq, SQ_E5) }); //min distance of king to centre squares.
    return ((4 - distanceToCentre)*(isEndgame(pos))); //bonus for king activity given in endgame based on distance to centre.
  }

  ///Piece Co-ordination
  //Co-ordinated pair of rooks
  //rooks on the same rank & file defend each other.
  float rooksCoordination(const Position& pos, Color c) {
    float score = 0;
    Bitboard myRooks = pos.pieces(c, ROOK);
    while (myRooks) {
        Square rookSq = pop_lsb(&myRooks);
        if (popcount(myRooks & file_bb(rookSq)))
            score += 8; //bonus if rooks on same file
        if (popcount(myRooks & rank_bb(rookSq)))
            score += 8; //bonus if rooks on same rank
    }
    return score;
  }
  //Bishops & Knights covering same squares
  //common combination useful for attack tactics.
  float othersCoordination(const Position& pos, Color c) {
    float score = 0;
    Bitboard myKnights = pos.pieces(c, KNIGHT);
    Bitboard myBishops = pos.pieces(c, BISHOP);
    while (myKnights) {
        Square knightSq = pop_lsb(&myKnights);
        score += 4*popcount(attacks_bb<BISHOP>(knightSq, myBishops)); //bonus
    }
    return score;
  }
  //King Protection
  float kingsCoordination(const Position& pos, Color c) {
    float score = 0;
    Square kingSq = pos.square<KING>(c);
    Bitboard surroundingSquares = pos.attacks_from<KING>(kingSq);
    score += 3*popcount(surroundingSquares & (pos.pieces(c, KNIGHT) | pos.pieces(c, BISHOP) | pos.pieces(c, ROOK)));
    //bonus for pieces covering square around king
    return score;
  }
  //Fiancetto Bishops
  //a common pattern of development which puts bishops on a long diagonal (covering many squares)
  //hard-coded.
  float fianchettoBishops(const Position& pos, Color c) {
    float score = 0;
    Bitboard myBishops = pos.pieces(c, BISHOP);
    const Bitboard fianchettoSquaresWhite = Bitboard(1ULL << SQ_G2) | Bitboard(1ULL << SQ_B2);
    const Bitboard fianchettoSquaresBlack = Bitboard(1ULL << SQ_G7) | Bitboard(1ULL << SQ_B7);
    if (c == WHITE && (myBishops & fianchettoSquaresWhite)) {
        if ((pos.pieces(WHITE, PAWN) & (Bitboard(1ULL << SQ_F2) | Bitboard(1ULL << SQ_H2))) && (pos.pieces(WHITE, PAWN) & (Bitboard(1ULL << SQ_F3) | Bitboard(1ULL << SQ_H3))))
            score += 3;
    }
    else if (c == BLACK && (myBishops & fianchettoSquaresBlack)) {
        if ((pos.pieces(BLACK, PAWN) & (Bitboard(1ULL << SQ_F7) | Bitboard(1ULL << SQ_H7))) && (pos.pieces(BLACK, PAWN) & (Bitboard(1ULL << SQ_F6) | Bitboard(1ULL << SQ_H6))))
            score += 3; //bonus if fiancetto bishop
    }
    return score;
  }
  ///Development
  //we want to develop knights and bishops as soon as possible.
  float developPieces(const Position& pos, Color c) {
    float score = 0;
    Bitboard undevelopedKnights = pos.pieces(c, KNIGHT) & (c == WHITE ? (Bitboard(1ULL << SQ_B1) | Bitboard(1ULL << SQ_G1)) : (Bitboard(1ULL << SQ_B8) | Bitboard(1ULL << SQ_G8)));
    Bitboard undevelopedBishops = pos.pieces(c, BISHOP) & (c == WHITE ? (Bitboard(1ULL << SQ_C1) | Bitboard(1ULL << SQ_F1)) : (Bitboard(1ULL << SQ_C8) | Bitboard(1ULL << SQ_F8)));
    score -= 3*popcount(undevelopedKnights); //if knight or bishop on starting square, apply penalty.
    score -= 3*popcount(undevelopedBishops);
    return score;
  }

  ///End of Features

  ///The evaluation function. The combination of weights and features.
  ///Since chess is zero-sum, we calculate each feature via (feature(white)-feature(black)), apart from first few where this is built in.

  Value Evaluation::value() {
 //   assert(!pos.checkers()); ///Disabled for training.
    int vv =
    //Material
        + pawnWeight*pawnScore(pos)
        + knightWeight*knightScore(pos)
        + bishopWeight*bishopScore(pos)
        + rookWeight*rookScore(pos)
    	+ queenWeight*queenScore(pos)
    	+ current_weights[0]*bishopPair(pos)
        //King Safety
        + current_weights[1] * (kingShelter(pos,WHITE) - kingShelter(pos,BLACK))
        + current_weights[2] * (pawnStorm(pos,WHITE) - pawnStorm(pos,BLACK))
        + current_weights[3] * (kingEscapeRoutes(pos,WHITE) - kingEscapeRoutes(pos,BLACK))
        + current_weights[4] * (kingTropism(pos,WHITE) - kingTropism(pos,BLACK))
        //Pawn Structure
        + current_weights[5] * (doubledPawns(pos,WHITE) - doubledPawns(pos,BLACK))
        + current_weights[6] * (isolatedPawns(pos,WHITE) - isolatedPawns(pos,BLACK))
        + current_weights[7] * (passedPawns(pos,WHITE) - passedPawns(pos,BLACK))
        + current_weights[8] * (rooksBehindPP(pos,WHITE) - rooksBehindPP(pos,BLACK))
        + current_weights[9] * (backwardPawns(pos,WHITE) - backwardPawns(pos,BLACK))
        + current_weights[10] * (pawnChains(pos,WHITE) - pawnChains(pos,BLACK))
        + current_weights[11] * (pawnMobility(pos,WHITE) - pawnMobility(pos,BLACK))
        + current_weights[12] * (pawnPhalanx(pos,WHITE) - pawnPhalanx(pos,BLACK))
        + current_weights[13] * (pawnOutposts(pos,WHITE) - pawnOutposts(pos,BLACK))
        //Piece Mobility
        + current_weights[14] * (knightMobility(pos,WHITE) - knightMobility(pos,BLACK))
        + current_weights[15] * (bishopMobility(pos,WHITE) - bishopMobility(pos,BLACK))
        + current_weights[16] * (rookMobility(pos,WHITE) - rookMobility(pos,BLACK))
        + current_weights[17] * (queenMobility(pos,WHITE) - queenMobility(pos,BLACK))
        + current_weights[18] * (centralControl(pos,WHITE) - centralControl(pos,BLACK))
        + current_weights[19] * (rookOnFile(pos,WHITE) - rookOnFile(pos,BLACK))
        + current_weights[20] * (queenOnFile(pos,WHITE) - queenOnFile(pos,BLACK))
        //Trapped Pieces
        + current_weights[21] * trappedBishop(pos)
        + current_weights[22] * (rookTrappedByKing(pos,WHITE) - rookTrappedByKing(pos,BLACK))
        //Tactics
        + current_weights[23] * (havePins(pos,WHITE) - havePins(pos,BLACK))
        + current_weights[24] * (haveForks(pos,WHITE) - haveForks(pos,BLACK))
        + current_weights[25] * (haveSkewers(pos,WHITE) - haveSkewers(pos,BLACK))
        + current_weights[26] * (discoveredAttacks(pos,WHITE) - discoveredAttacks(pos,BLACK))
        + current_weights[27] * (doubleAttacks(pos,WHITE) - doubleAttacks(pos,BLACK))
        + current_weights[28] * (haveOverloads(pos,WHITE) - haveOverloads(pos,BLACK))
        + current_weights[29] * (hangingPieces(pos,WHITE) - hangingPieces(pos,BLACK))
        //Endgame
        + current_weights[30]* (egkingActivity(pos,WHITE) - egkingActivity(pos,BLACK))
        //Piece Co-ordination
        + current_weights[31] * (rooksCoordination(pos,WHITE) - rooksCoordination(pos,BLACK))
        + current_weights[32] * (othersCoordination(pos,WHITE) - othersCoordination(pos,BLACK))
        + current_weights[33] * (kingsCoordination(pos,WHITE) - kingsCoordination(pos,BLACK))
        + current_weights[34] * (fianchettoBishops(pos,WHITE) - fianchettoBishops(pos,BLACK))
        //Development
        + current_weights[35] * (developPieces(pos,WHITE) - developPieces(pos,BLACK));

        vv*=PawnValueEg;
        vv*=0.01;

    return  Value(pos.side_to_move() == WHITE ? vv : -vv);
  }
} // namespace


/// evaluate() is the evaluator for the outer world. It returns a static evaluation of the position from the point of view of the side to move
Value Eval::evaluate(const Position& pos) {
  return Evaluation(pos).value();
}


/// trace() returns a string with detailed breakdowns of evaluation terms, aiding in debugging, unlike evaluate(), which returns a numerical value
std::string Eval::trace(const Position& pos) {
  if (pos.checkers())
      return "Total evaluation: none (in check)";
  Value v = Evaluation(pos).value();
  v = pos.side_to_move() == WHITE ? v : -v;// Trace scores are from white's point of view
  std::stringstream ss;
  ss << "Final evaluation: " << double(v) / PawnValueEg << " (white side)\n";
  return ss.str();
}
}
