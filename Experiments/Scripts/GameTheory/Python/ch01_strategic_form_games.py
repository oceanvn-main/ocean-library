"""
Chapter 1: Games in Strategic Form and Nash Equilibrium
======================================================
Based on Fudenberg & Tirole (1991), 'Game Theory', Chapter 1.

This module implements classic strategic-form game examples from Chapter 1 using 
Google DeepMind's OpenSpiel (pyspiel) framework and game-theoretic solvers.

Examples Included:
  1. Prisoner's Dilemma (Strict Dominance & Pareto Inefficiency)
  2. Battle of the Sexes (Multiple Pure NE & Mixed Strategy NE)
  3. Matching Pennies (Zero-Sum Game & Mixed Strategy NE)
  4. Discretized Cournot Duopoly (Iterated Strict Dominance & Best Response Dynamics)
  5. 3-Player Prisoner's Dilemma (N-Player Strategic Form & 2-Matrix Textbook Layout)
"""

import sys
import numpy as np
import pyspiel
import nashpy as nash


def create_open_spiel_matrix_game(title: str, short_name: str, 
                                  row_actions: list, col_actions: list, 
                                  row_payoffs: np.ndarray, col_payoffs: np.ndarray) -> pyspiel.Game:
    """
    Constructs an OpenSpiel MatrixGame object from 2-player payoff matrices.
    """
    return pyspiel.create_matrix_game(
        short_name,
        title,
        row_actions,
        col_actions,
        row_payoffs.tolist(),
        col_payoffs.tolist()
    )


def create_open_spiel_tensor_game(title: str, short_name: str,
                                  player_actions: list,
                                  player_payoffs: list) -> pyspiel.Game:
    """
    Constructs an OpenSpiel TensorGame object for N-player games.
    """
    return pyspiel.create_tensor_game(
        short_name,
        title,
        player_actions,
        player_payoffs
    )


def print_textbook_matrix_2p(title: str, row_actions: list, col_actions: list, 
                             row_payoffs: np.ndarray, col_payoffs: np.ndarray):
    """
    Prints a clean 2D textbook payoff matrix table format.
    Each cell displays (u1, u2).
    """
    print(f"\n--- {title} (Textbook Payoff Matrix Table) ---")
    header = "               " + "".join([f"{act:^16}" for act in col_actions])
    print(header)
    print("         " + "-" * (len(header) - 9))
    for i, r_act in enumerate(row_actions):
        row_str = f"{r_act:>8} | "
        for j, c_act in enumerate(col_actions):
            p1 = row_payoffs[i, j]
            p2 = col_payoffs[i, j]
            val = f"({p1:g}, {p2:g})"
            row_str += f"{val:^16}"
        print(row_str)


def print_textbook_matrix_3p(title: str, row_actions: list, col_actions: list, layer_actions: list,
                             payoffs_3d: np.ndarray):
    """
    Prints a 3-player game as TWO separate 2D textbook matrices,
    one for each action of Player 3 (e.g. Matrix 1 when Player 3 = Action 1, Matrix 2 when Player 3 = Action 2).
    Each cell displays (u1, u2, u3).
    """
    print(f"\n--- {title} (Textbook 3-Player 2-Matrix Presentation) ---")
    n_r, n_c, n_l, _ = payoffs_3d.shape

    for k, l_act in enumerate(layer_actions):
        print(f"\n  [ MATRIX {k+1}: Player 3 chooses '{l_act}' ]")
        header = "               " + "".join([f"{act:^18}" for act in col_actions])
        print("                Player 2 actions:")
        print(header)
        print("         " + "-" * (len(header) - 9))
        for i, r_act in enumerate(row_actions):
            row_str = f"{r_act:>8} | "
            for j, c_act in enumerate(col_actions):
                u1, u2, u3 = payoffs_3d[i, j, k]
                val = f"({u1:g}, {u2:g}, {u3:g})"
                row_str += f"{val:^18}"
            print(row_str)


def compute_expected_payoffs(row_payoffs: np.ndarray, col_payoffs: np.ndarray, 
                             row_strat: np.ndarray, col_strat: np.ndarray) -> tuple:
    """
    Computes expected payoffs u_1(sigma_1, sigma_2) and u_2(sigma_1, sigma_2)
    given mixed strategy probability vectors sigma_1 and sigma_2.
    """
    u1 = float(row_strat @ row_payoffs @ col_strat)
    u2 = float(row_strat @ col_payoffs @ col_strat)
    return u1, u2


def find_strictly_dominated_strategies(row_payoffs: np.ndarray, col_payoffs: np.ndarray,
                                       row_actions: list, col_actions: list):
    """
    Identifies pure strategies that are strictly dominated by another pure strategy.
    """
    n_rows, n_cols = row_payoffs.shape
    dominated_row = []
    dominated_col = []

    # Player 1 (Row Player): strategy i is strictly dominated by i' if u1(i', j) > u1(i, j) for all j
    for i in range(n_rows):
        for i_prime in range(n_rows):
            if i != i_prime and np.all(row_payoffs[i_prime, :] > row_payoffs[i, :]):
                dominated_row.append((row_actions[i], row_actions[i_prime]))
                break

    # Player 2 (Col Player): strategy j is strictly dominated by j' if u2(i, j') > u2(i, j) for all i
    for j in range(n_cols):
        for j_prime in range(n_cols):
            if j != j_prime and np.all(col_payoffs[:, j_prime] > col_payoffs[:, j]):
                dominated_col.append((col_actions[j], col_actions[j_prime]))
                break

    return dominated_row, dominated_col


def iterated_elimination_of_dominated_strategies(row_payoffs: np.ndarray, col_payoffs: np.ndarray,
                                                 row_actions: list, col_actions: list):
    """
    Performs Iterated Elimination of Strictly Dominated Strategies (IESDS).
    Returns history of eliminated strategies and the reduced game matrix.
    """
    curr_r_payoffs = row_payoffs.copy()
    curr_c_payoffs = col_payoffs.copy()
    curr_r_actions = list(row_actions)
    curr_c_actions = list(col_actions)
    elimination_steps = []

    changed = True
    step_num = 1
    while changed:
        changed = False
        n_rows, n_cols = curr_r_payoffs.shape

        # Check Player 1 (Row)
        for i in range(n_rows):
            for i_prime in range(n_rows):
                if i != i_prime and np.all(curr_r_payoffs[i_prime, :] > curr_r_payoffs[i, :]):
                    elim_act = curr_r_actions[i]
                    dom_act = curr_r_actions[i_prime]
                    elimination_steps.append((step_num, "Player 1", elim_act, dom_act))
                    curr_r_payoffs = np.delete(curr_r_payoffs, i, axis=0)
                    curr_c_payoffs = np.delete(curr_c_payoffs, i, axis=0)
                    curr_r_actions.pop(i)
                    changed = True
                    step_num += 1
                    break
            if changed:
                break

        if changed:
            continue

        # Check Player 2 (Col)
        for j in range(n_cols):
            for j_prime in range(n_cols):
                if j != j_prime and np.all(curr_c_payoffs[:, j_prime] > curr_c_payoffs[:, j]):
                    elim_act = curr_c_actions[j]
                    dom_act = curr_c_actions[j_prime]
                    elimination_steps.append((step_num, "Player 2", elim_act, dom_act))
                    curr_r_payoffs = np.delete(curr_r_payoffs, j, axis=1)
                    curr_c_payoffs = np.delete(curr_c_payoffs, j, axis=1)
                    curr_c_actions.pop(j)
                    changed = True
                    step_num += 1
                    break
            if changed:
                break

    return elimination_steps, curr_r_actions, curr_c_actions, curr_r_payoffs, curr_c_payoffs


# =====================================================================
# EXAMPLE 1: Prisoner's Dilemma
# =====================================================================
def run_prisoners_dilemma():
    print("\n" + "="*70)
    print("EXAMPLE 1: PRISONER'S DILEMMA (Fudenberg & Tirole Section 1.1)")
    print("="*70)

    row_actions = ["Defect (D)", "Cooperate (C)"]
    col_actions = ["Defect (D)", "Cooperate (C)"]
    
    # Payoff matrices u_1 and u_2
    row_payoffs = np.array([
        [-2.0,  0.0],
        [-3.0, -1.0]
    ])
    col_payoffs = np.array([
        [-2.0, -3.0],
        [ 0.0, -1.0]
    ])

    # Textbook Matrix Display
    print_textbook_matrix_2p("Prisoner's Dilemma", row_actions, col_actions, row_payoffs, col_payoffs)

    # 1. Create OpenSpiel Game
    game = create_open_spiel_matrix_game("Prisoner's Dilemma", "prisoners_dilemma",
                                          row_actions, col_actions,
                                          row_payoffs, col_payoffs)
    state = game.new_initial_state()
    print("\nOpenSpiel State Representation:")
    print(state)

    # 2. Dominance Analysis
    dom_r, dom_c = find_strictly_dominated_strategies(row_payoffs, col_payoffs, row_actions, col_actions)
    print("\nDominance Analysis:")
    for elim, dom in dom_r:
        print(f"  - Player 1 strategy '{elim}' is STRICTLY DOMINATED by '{dom}'")
    for elim, dom in dom_c:
        print(f"  - Player 2 strategy '{elim}' is STRICTLY DOMINATED by '{dom}'")

    # 3. Solve Nash Equilibrium via NashPy
    nash_game = nash.Game(row_payoffs, col_payoffs)
    equilibria = list(nash_game.support_enumeration())

    print("\nNash Equilibria:")
    for idx, (sigma1, sigma2) in enumerate(equilibria, 1):
        u1, u2 = compute_expected_payoffs(row_payoffs, col_payoffs, sigma1, sigma2)
        print(f"  NE #{idx}: Player 1 profile = {sigma1}, Player 2 profile = {sigma2}")
        print(f"       Expected Utilities: u1 = {u1:.2f}, u2 = {u2:.2f}")

    # 4. Pareto Inefficiency Demonstration
    u1_coop, u2_coop = compute_expected_payoffs(row_payoffs, col_payoffs, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    u1_ne, u2_ne = compute_expected_payoffs(row_payoffs, col_payoffs, equilibria[0][0], equilibria[0][1])
    print("\nPareto Analysis:")
    print(f"  - Nash Equilibrium (D, D) payoff: ({u1_ne}, {u2_ne})")
    print(f"  - Mutual Cooperation (C, C) payoff: ({u1_coop}, {u2_coop})")
    print("  -> (C, C) strictly Pareto-dominates (D, D), illustrating the classic dilemma.")


# =====================================================================
# EXAMPLE 2: Battle of the Sexes (Bach or Stravinsky)
# =====================================================================
def run_battle_of_the_sexes():
    print("\n" + "="*70)
    print("EXAMPLE 2: BATTLE OF THE SEXES / BACH OR STRAVINSKY (Section 1.2)")
    print("="*70)

    row_actions = ["Opera (B)", "Football (S)"]
    col_actions = ["Opera (B)", "Football (S)"]

    row_payoffs = np.array([
        [2.0, 0.0],
        [0.0, 1.0]
    ])
    col_payoffs = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ])

    print_textbook_matrix_2p("Battle of the Sexes", row_actions, col_actions, row_payoffs, col_payoffs)

    game = create_open_spiel_matrix_game("Battle of the Sexes", "battle_of_sexes",
                                          row_actions, col_actions,
                                          row_payoffs, col_payoffs)

    nash_game = nash.Game(row_payoffs, col_payoffs)
    equilibria = list(nash_game.support_enumeration())

    print("\nComputed Nash Equilibria:")
    for idx, (sigma1, sigma2) in enumerate(equilibria, 1):
        u1, u2 = compute_expected_payoffs(row_payoffs, col_payoffs, sigma1, sigma2)
        eq_type = "Pure Strategy" if (np.max(sigma1) == 1.0 and np.max(sigma2) == 1.0) else "Mixed Strategy"
        print(f"  NE #{idx} ({eq_type}):")
        print(f"    Player 1 strategy: {sigma1}  [P(Opera) = {sigma1[0]:.4f}, P(Football) = {sigma1[1]:.4f}]")
        print(f"    Player 2 strategy: {sigma2}  [P(Opera) = {sigma2[0]:.4f}, P(Football) = {sigma2[1]:.4f}]")
        print(f"    Expected Utilities: u1 = {u1:.4f}, u2 = {u2:.4f}")


# =====================================================================
# EXAMPLE 3: Matching Pennies
# =====================================================================
def run_matching_pennies():
    print("\n" + "="*70)
    print("EXAMPLE 3: MATCHING PENNIES (Zero-Sum Game) (Section 1.2)")
    print("="*70)

    row_actions = ["Heads (H)", "Tails (T)"]
    col_actions = ["Heads (H)", "Tails (T)"]

    row_payoffs = np.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ])
    col_payoffs = -row_payoffs

    print_textbook_matrix_2p("Matching Pennies", row_actions, col_actions, row_payoffs, col_payoffs)

    game = create_open_spiel_matrix_game("Matching Pennies", "matching_pennies",
                                          row_actions, col_actions,
                                          row_payoffs, col_payoffs)

    nash_game = nash.Game(row_payoffs, col_payoffs)
    equilibria = list(nash_game.support_enumeration())

    print("\nComputed Nash Equilibria:")
    for idx, (sigma1, sigma2) in enumerate(equilibria, 1):
        u1, u2 = compute_expected_payoffs(row_payoffs, col_payoffs, sigma1, sigma2)
        print(f"  NE #{idx}:")
        print(f"    Player 1 strategy: {sigma1}")
        print(f"    Player 2 strategy: {sigma2}")
        print(f"    Expected Utilities (Value of Game): u1 = {u1:.4f}, u2 = {u2:.4f}")


# =====================================================================
# EXAMPLE 4: Discretized Cournot Duopoly Competition
# =====================================================================
def run_cournot_duopoly(market_demand_a=12.0, marginal_cost_c=0.0, max_qty=6):
    print("\n" + "="*70)
    print("EXAMPLE 4: DISCRETIZED COURNOT DUOPOLY COMPETITION (Section 1.2)")
    print("="*70)

    quantities = list(range(max_qty + 1))
    row_actions = [f"q1={q}" for q in quantities]
    col_actions = [f"q2={q}" for q in quantities]
    n_q = len(quantities)

    row_payoffs = np.zeros((n_q, n_q))
    col_payoffs = np.zeros((n_q, n_q))

    for i, q1 in enumerate(quantities):
        for j, q2 in enumerate(quantities):
            total_q = q1 + q2
            price = max(market_demand_a - total_q, 0.0)
            row_payoffs[i, j] = q1 * (price - marginal_cost_c)
            col_payoffs[i, j] = q2 * (price - marginal_cost_c)

    print_textbook_matrix_2p("Cournot Duopoly (7x7 Payoff Matrix)", row_actions, col_actions, row_payoffs, col_payoffs)

    game = create_open_spiel_matrix_game("Cournot Duopoly", "cournot_duopoly",
                                          row_actions, col_actions,
                                          row_payoffs, col_payoffs)

    # Iterated Elimination of Strictly Dominated Strategies (IESDS)
    print("\nIterated Elimination of Strictly Dominated Strategies (IESDS):")
    steps, r_rem_act, c_rem_act, r_rem_mat, c_rem_mat = iterated_elimination_of_dominated_strategies(
        row_payoffs, col_payoffs, row_actions, col_actions
    )
    if steps:
        for s_num, player, elim_act, dom_act in steps:
            print(f"  Step {s_num}: {player} strategy '{elim_act}' eliminated (dominated by '{dom_act}')")

    nash_game = nash.Game(row_payoffs, col_payoffs)
    equilibria = list(nash_game.support_enumeration())

    print("\nComputed Discrete Nash Equilibrium:")
    for idx, (sigma1, sigma2) in enumerate(equilibria, 1):
        q1_eq = float(np.dot(sigma1, quantities))
        q2_eq = float(np.dot(sigma2, quantities))
        u1_eq, u2_eq = compute_expected_payoffs(row_payoffs, col_payoffs, sigma1, sigma2)
        print(f"  NE #{idx}:")
        print(f"    Player 1 profile: q1 = {q1_eq:.2f}, Expected Utility = {u1_eq:.2f}")
        print(f"    Player 2 profile: q2 = {q2_eq:.2f}, Expected Utility = {u2_eq:.2f}")


# =====================================================================
# EXAMPLE 5: 3-Player Prisoner's Dilemma / Public Goods Game
# =====================================================================
def run_three_player_game():
    print("\n" + "="*70)
    print("EXAMPLE 5: 3-PLAYER PRISONER'S DILEMMA (TEXTBOOK 2-MATRIX LAYOUT)")
    print("="*70)
    print("In textbook game theory, 3-Player games are presented as TWO 2D matrices:")
    print("  - Matrix 1: Payoffs when Player 3 plays 'Cooperate (C)'")
    print("  - Matrix 2: Payoffs when Player 3 plays 'Defect (D)'")

    player1_actions = ["Cooperate (C)", "Defect (D)"]
    player2_actions = ["Cooperate (C)", "Defect (D)"]
    player3_actions = ["Cooperate (C)", "Defect (D)"]

    # 3D payoff tensor: shape (2, 2, 2, 3) -> (P1_action, P2_action, P3_action, player_payoffs)
    payoffs_3d = np.zeros((2, 2, 2, 3))

    # --- Matrix 1: Player 3 plays Cooperate (C) ---
    # (C, C, C) -> (2, 2, 2)
    payoffs_3d[0, 0, 0] = [2.0, 2.0, 2.0]
    # (C, D, C) -> (1, 3, 1)
    payoffs_3d[0, 1, 0] = [1.0, 3.0, 1.0]
    # (D, C, C) -> (3, 1, 1)
    payoffs_3d[1, 0, 0] = [3.0, 1.0, 1.0]
    # (D, D, C) -> (2, 2, 0)
    payoffs_3d[1, 1, 0] = [2.0, 2.0, 0.0]

    # --- Matrix 2: Player 3 plays Defect (D) ---
    # (C, C, D) -> (1, 1, 3)
    payoffs_3d[0, 0, 1] = [1.0, 1.0, 3.0]
    # (C, D, D) -> (0, 2, 2)
    payoffs_3d[0, 1, 1] = [0.0, 2.0, 2.0]
    # (D, C, D) -> (2, 0, 2)
    payoffs_3d[1, 0, 1] = [2.0, 0.0, 2.0]
    # (D, D, D) -> (1, 1, 1)
    payoffs_3d[1, 1, 1] = [1.0, 1.0, 1.0]

    # Print Textbook 2-Matrix Presentation
    print_textbook_matrix_3p("3-Player Prisoner's Dilemma", 
                            player1_actions, player2_actions, player3_actions, 
                            payoffs_3d)

    # Build OpenSpiel TensorGame
    # Flatten payoff arrays for each player: S1 x S2 x S3 = 2 x 2 x 2 = 8 entries
    u0 = payoffs_3d[:, :, :, 0].flatten()
    u1 = payoffs_3d[:, :, :, 1].flatten()
    u2 = payoffs_3d[:, :, :, 2].flatten()

    tensor_game = create_open_spiel_tensor_game(
        "3-Player Prisoner's Dilemma", "3p_prisoners_dilemma",
        [player1_actions, player2_actions, player3_actions],
        [u0, u1, u2]
    )

    print("\nOpenSpiel 3-Player TensorGame Created:")
    print(f"  Game Type: {tensor_game.get_type().long_name}")
    print(f"  Num Players: {tensor_game.num_players()}")

    # Dominance & Equilibrium Analysis
    print("\nGame Analysis:")
    print("  - Strict Dominance: Defect (D) strictly dominates Cooperate (C) for ALL 3 players.")
    print("  - Unique Nash Equilibrium: (D, D, D) with payoff (1, 1, 1).")
    print("  - Mutual Cooperation Outcome: (C, C, C) with payoff (2, 2, 2).")
    print("  -> Mutual Cooperation strictly Pareto-dominates the Nash Equilibrium (1, 1, 1).")


def main():
    print("="*70)
    print(" GAME THEORY CHAPTER 1: STRATEGIC-FORM GAMES & NASH EQUILIBRIUM ")
    print(" OpenSpiel & Game-Theoretic Solvers Demo ")
    print(" Reference: Fudenberg & Tirole (1991)")
    print("="*70)

    run_prisoners_dilemma()
    run_battle_of_the_sexes()
    run_matching_pennies()
    run_cournot_duopoly(market_demand_a=12.0, marginal_cost_c=0.0, max_qty=6)
    run_three_player_game()

    print("\n" + "="*70)
    print(" Execution Completed Successfully! ")
    print("="*70)


if __name__ == "__main__":
    main()
