import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # d100 Probability Estimation

        This notebook estimates probabilities of successes in opposed skill tests of d100 combat in Mythras.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    import marimo as mo
    from enum import Enum
    from dataclasses import dataclass

    np.set_printoptions(linewidth=180)
    return Enum, dataclass, mo, np, plt, sns


@app.cell
def _(Enum, np):
    n = 10_000

    class CheckResult(Enum):
        FUMBLE = 0
        FAIL = 1
        SUCCESS = 2
        CRITICAL = 3    

    FUMBLE = 0
    FAIL = 1
    SUCCESS = 2
    CRITICAL = 3
    POSSIBLE_OUTCOMES = [FUMBLE, FAIL, SUCCESS, CRITICAL]

    def roll(n=1, sides=100):
      return np.random.randint(1, 1 + sides, n)
    return (
        CRITICAL,
        CheckResult,
        FAIL,
        FUMBLE,
        POSSIBLE_OUTCOMES,
        SUCCESS,
        n,
        roll,
    )


@app.cell
def _(CRITICAL, FAIL, FUMBLE, SUCCESS, np):
    # the roll of a d100 is a uniform probability
    possible_d100_rolls = np.arange(1, 101)

    attacker_skill = np.arange(10, 101, 10)
    defender_skill = np.arange(10, 101, 10)

    def compute_roll_outcomes(skills: np.ndarray):
        """
        Given an array of skill values, compute their success level for d100 outcome.

        :param skills: all the skill values to consider. 
            numpy array (num_skill_values)
        :return: a matrix (num_skill_values, 100)
        """
        # outcomes is (num_skill_values, 100)
        outcomes = np.full([len(skills), 100], FAIL, dtype=np.int8)

        inds = skills.reshape(-1, 1) >= possible_d100_rolls
        outcomes[inds] = SUCCESS

        # 01-05 is always a success
        outcomes[:, :5] = SUCCESS
        
        # 96-00 is always a fail
        outcomes[:, -5:] = FAIL
    
        crit_threshold = np.ceil(skills / 10).astype(int)
        fumble_threshold = np.where(skills >= 100, 100, 99)

        # if there's a way to vectorize this, it is too complicated
        for i in range(len(skills)):

            threshold = crit_threshold[i]
            outcomes[i, :threshold] = CRITICAL

            # -1 because we are changing die values to indices (crits are inclusive)
            threshold = fumble_threshold[i] - 1
            outcomes[i, threshold:] = FUMBLE

        return outcomes
    return (
        attacker_skill,
        compute_roll_outcomes,
        defender_skill,
        possible_d100_rolls,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## RAW 

        This section considers RAW from the Mythras books.

        ### Hit Probability

        Compute the probability of an attack successfully causing damage (before taking armor into account).

        This assumes the defender has a weapon able to fully block the damage of the attacker in case of a successful defense roll.
        """
    )
    return


@app.cell
def _(CRITICAL, SUCCESS, compute_roll_outcomes, np):
    def compute_hit_matrix_raw(attacker_skill: np.ndarray, defender_skill: np.ndarray) -> np.ndarray:
        """
        Compute a hit matrix of the chance of landing a blow according to RAW.

        It assumes a defender success fully parries the damage.

        :param attacker_skill: an array (num_skill_values) with all skill values to consider
        :param defender_skill: same as above, for the defender
        :return: an array (num_attacker_values, num_defender_values) with the probability that
            each combination of skill values results in a successful blow.
        """
        attacker_outcomes = compute_roll_outcomes(attacker_skill)
        defender_outcomes = compute_roll_outcomes(defender_skill)
    
        # all possible outcomes (num_skill_values, 100)
        any_attacker_success = (attacker_outcomes == SUCCESS) | (attacker_outcomes == CRITICAL)
        any_defender_success = (defender_outcomes == SUCCESS) | (defender_outcomes == CRITICAL)
    
        # these are (num_skill_values,) probabilities
        p_attacker_success = any_attacker_success.mean(1, keepdims=True)
        p_defender_success = any_defender_success.mean(1, keepdims=True)
    
        # compute the probability that an attack will hit (won't be parried)
        hit_matrix = p_attacker_success * (1 - p_defender_success.T)
    
        return hit_matrix

    return (compute_hit_matrix_raw,)


@app.cell
def _(attacker_skill, compute_hit_matrix_raw, defender_skill, plt, sns):
    hit_matrix = compute_hit_matrix_raw(attacker_skill, defender_skill)

    # turn it into percentages
    hit_pct = hit_matrix * 100

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    flat_att_skill = attacker_skill.flatten()
    flat_def_skill = defender_skill.flatten()

    plot = sns.heatmap(
        data=hit_pct, annot=True, xticklabels=flat_att_skill, yticklabels=flat_def_skill
    )


    plot.set(
        title="Hit Chance %",
        xlabel="Defense Skill",
        ylabel="Attack Skill",
    )
    plot
    return flat_att_skill, flat_def_skill, hit_matrix, hit_pct, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Special Effects

        Compute the probabilities of causing Special Effects
        """
    )
    return


@app.cell
def _(CRITICAL, POSSIBLE_OUTCOMES, SUCCESS, compute_roll_outcomes, np):
    def get_result_probability(outcomes: np.ndarray, success_level: int):
        """
        Compute the probability of a certain outcome for a series of skill values.

        :param outcomes: array (num_skill_values, 100) indicating the outcome for the d100 for each skill value
        :param success_level: integer representation of the success levels
        :return: an array (num_skill_values, 1) of probabilities
        """
        return (outcomes == success_level).mean(1, keepdims=True)

    def compute_expected_se(attacker_skill, defender_skill, at_least_one=False):
        """
        Compute the expected number of SEs according to RAW.

        :param at_least_one: if True, only consider the chance of causing at least one effect.
            If False, compute the expected total SEs.
        """
        attacker_outcomes = compute_roll_outcomes(attacker_skill)
        defender_outcomes = compute_roll_outcomes(defender_skill)
    
        # outcomes is (num_values, 100)
        p_attacker_crit = get_result_probability(attacker_outcomes, CRITICAL)
        p_attacker_normal = get_result_probability(attacker_outcomes, SUCCESS)
    
        p_defender_crit = get_result_probability(defender_outcomes, CRITICAL)
        p_defender_normal = get_result_probability(defender_outcomes, SUCCESS)
    
        p_attacker_outcomes = {}
        p_defender_outcomes = {}
    
        for outcome in POSSIBLE_OUTCOMES:
            p_attacker_outcomes[outcome] = get_result_probability(attacker_outcomes, outcome)
            p_defender_outcomes[outcome] = get_result_probability(defender_outcomes, outcome)
            expected_num_se = np.zeros([len(attacker_skill), len(defender_skill)])
        
        for att_outcome in [SUCCESS, CRITICAL]:
            # p_att_outcome is an array (num_skill_values, 1)
            p_att_outcome = p_attacker_outcomes[att_outcome]
    
            for def_outcome in POSSIBLE_OUTCOMES:
                # consider only attacker SE's
                if def_outcome >= att_outcome:
                    continue
            
                # p_def_outcome is an array (num_skill_values, 1)
                p_def_outcome = p_defender_outcomes[def_outcome]
    
                # combination probability of the attack and defense outcomes
                p_combination = p_att_outcome * p_def_outcome.T

                if at_least_one:
                    num_se = 1
                else:
                    num_se = att_outcome - def_outcome
                    assert num_se > 0
    
                expected_num_se += num_se * p_combination

        return expected_num_se
    return compute_expected_se, get_result_probability


@app.cell
def _(
    attacker_skill,
    compute_expected_se,
    defender_skill,
    flat_att_skill,
    flat_def_skill,
    mo,
    plt,
    sns,
):
    expected_num_se = compute_expected_se(attacker_skill, defender_skill)
    at_least_one_se = compute_expected_se(attacker_skill, defender_skill, at_least_one=True)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))
    plot_se = sns.heatmap(
        data=expected_num_se, annot=True, xticklabels=flat_att_skill, yticklabels=flat_def_skill
    )
    fig = plt.figure()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))
    plot_se_1 = sns.heatmap(
        data=at_least_one_se, annot=True, xticklabels=flat_att_skill, yticklabels=flat_def_skill
    )

    # obs = '''The percentages refer to the point of view of the attacker. For the defender, the values are the same with the axes swapped.'''
    # plt.annotate(obs, xy=(0.5, 0.5), xytext=(1.5, 1.5), wrap=True,
    #              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5) 
    #             )

    plot_se.set(
        title="Expected Special Effects",
        xlabel="Defense Skill",
        ylabel="Attack Skill",
    )

    plot_se_1.set(
        title="At least one SE %",
        xlabel="Defense Skill",
        ylabel="Attack Skill",
    )

    mo.hstack([plot_se, plot_se_1])
    return at_least_one_se, expected_num_se, fig, plot_se, plot_se_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Homebrew

        Main features:

        - Weak attacks

        - Less occurrences of multiple Special Effects

        - Opposed rolls instead of differential rolls

        Explanations:

        - For all purposes, defense fumbles count as normal failures (do not count an extra degree of success for the attacker).

            - **Rationale**: avoid sudden bursts of SEs that could easily finish a fight without feeling particularly realistic.

        - Attack fumbles causes the attack to miss.

        - Successful defenses against failed attacks cause only one SE (regardless of fumbles and criticals).

            - **Rationale**: penalize less the attacker and allow less advantages from "turtling".

        - Failed attacks are actually weak attacks that can cause a "weak hit"
            - **Rationale**: Completely missing the target is unlikely; the attacker should at least swing the weapon to the right direction.
            - Weak attacks are blocked by any successful defense. A successful defense also applies one Special Effect.
            - In case it hits, roll damage with disadvantage.

        - A successful attack can only be parried if the defender wins an opposed roll (i.e., the rolled value from the defender's roll is higher)

            - **Rationale**: avoid stalling combat at higher skill levels when most attacks result in misses.
            - In case the attacker is wins the opposed test but the defender succeeded in the test, cause damage but apply no SE.
            - In case the defender wins, the attack misses and the attacker suffers no SE.

        - A critical hit follow the same rationale as above (only parried by a critical defense that wins the opposed roll)

            - Against a failed defense, apply one normal SE plus one critical-only SE.
            - Against a critical defense that lost the opposed roll, only cause damage.

        """
    )
    return


@app.cell
def _(CheckResult, Enum, compute_roll_outcomes, dataclass, np):
    class HitType(Enum):
        MISS_WITH_DEF_SE = 0
        MISS = 1
        WEAK_HIT = 2
        STRONG_HIT = 3
        STRONG_HIT_WITH_SE = 4
        CRITICAL_HIT_WITH_SE = 5

    @dataclass
    class AttackOutcome:
        """
        Class to store probabilities of different types of attack outcomes.
        """
        p_miss_with_se: np.ndarray
        p_miss: np.ndarray
        p_weak_hit: np.ndarray
        p_strong_hit: np.ndarray
        p_strong_hit_with_se: np.ndarray
        p_crit_with_se: np.ndarray

        @property
        def p_any_hit(self):
            p = self.p_weak_hit + self.p_any_strong_hit
            return p

        @property
        def p_any_strong_hit(self):
            p = self.p_strong_hit + self.p_strong_hit_with_se + self.p_crit_with_se
            return p

        @property
        def p_attacker_se(self):
            return self.p_strong_hit_with_se + self.p_crit_with_se
    

    def compute_hit_matrix_hr(attacker_skill: np.ndarray, defender_skill: np.ndarray) -> AttackOutcome:
        """
        Compute a hit matrix of the chance of landing a blow according to house rules.

        It assumes a defender success fully parries the damage.

        :param attacker_skill: an array (num_skill_values) with all skill values to consider
        :param defender_skill: same as above, for the defender
        :return: an array (num_attacker_values, num_defender_values) with the probability that
            each combination of skill values results in a successful blow.
        """
        attacker_outcomes = compute_roll_outcomes(attacker_skill)
        defender_outcomes = compute_roll_outcomes(defender_skill)

        # (skill_values, 100)
        defender_success = defender_outcomes == CheckResult.SUCCESS.value
        defender_crit = defender_outcomes == CheckResult.CRITICAL.value

        any_defender_success = defender_success | defender_crit
        any_defender_fail = ~any_defender_success

        p_defender_success = defender_success.mean(1)
        p_defender_crit = defender_crit.mean(1)
        p_any_defender_fail = any_defender_fail.mean(1)
        p_any_defender_success = any_defender_success.mean(1)

        # 0. attacker fumbles - shape is (skill_values, 100)
        attacker_fumbles = attacker_outcomes == CheckResult.FUMBLE.value
        p_attacker_fumbles = attacker_fumbles.mean(1)

        p_miss = np.outer(p_attacker_fumbles, p_any_defender_fail)
        p_miss_with_se = np.outer(p_attacker_fumbles, p_any_defender_success)

    
        # 1. weak attacks (attack fail but not fumble) - shape is (skill_values, 100)
        weak_attack = attacker_outcomes == CheckResult.FAIL.value
        p_weak_attack = weak_attack.mean(1)
    
        p_weak_hit = np.outer(p_weak_attack, p_any_defender_fail)
        p_miss_with_se += np.outer(p_weak_attack, p_any_defender_success)


        # 2. strong attacks (compare numeric values) - shape is (skill_values, 100)
        attacker_success = attacker_outcomes == CheckResult.SUCCESS.value
        p_attacker_success = attacker_success.mean(1)

        # 2.1: defender fails
        p_strong_hit_with_se = np.outer(p_attacker_success, p_any_defender_fail)

        # 2.2: defender suceeds
    
        # opposed_dice[i, j] contains whether the attacker roll is higher than the defender roll
        # (not accounting for skill cap)
        opposed_dice = np.zeros([100, 100], dtype=np.bool)
    
        inds_i, inds_j = np.tril_indices(100, -1)
        opposed_dice[inds_i, inds_j] = True
        opposed_dice = opposed_dice.reshape(1, 100, 1, 100)

        m = len(attacker_skill)
        n = len(defender_skill)

        # 4d array containing whether both succeed at the d100 (att_skill, 100, def_skill, 100)
        both_success = np.outer(attacker_success, defender_success).reshape(m, 100, n, 100)
        attacker_blackjack_success = both_success * opposed_dice
        defender_blackjack_success = both_success * (~opposed_dice)

        p_strong_hit = attacker_blackjack_success.mean(3).mean(1)
        p_miss += defender_blackjack_success.mean(3).mean(1)

        # 2.3 defender crits
        p_miss_with_se += np.outer(p_attacker_success, p_defender_crit)

        # 3. attacker crits
        attacker_crit = attacker_outcomes == CheckResult.CRITICAL.value
        p_attacker_crit = attacker_crit.mean(1)

        # 3.1 defender fails
        p_crit_with_se = np.outer(p_attacker_crit, p_any_defender_fail)

        # 3.2 defender succeeds
        p_strong_hit_with_se += np.outer(p_attacker_crit, p_defender_success)

        # 3.3 defender crits

        # 4d array containing whether both crit at the d100 (att_skill, 100, def_skill, 100)
        both_crit = np.outer(attacker_crit, defender_crit).reshape(m, 100, n, 100)
        attacker_blackjack_crit = both_crit * opposed_dice
        defender_blackjack_crit = both_crit * (~opposed_dice)

        p_strong_hit += attacker_blackjack_crit.mean(3).mean(1)
        p_miss += defender_blackjack_crit.mean(3).mean(1)

        outcome = AttackOutcome(p_miss_with_se, p_miss, p_weak_hit, p_strong_hit, p_strong_hit_with_se, p_crit_with_se)

        return outcome
    return AttackOutcome, HitType, compute_hit_matrix_hr


@app.cell
def _(mo):
    att_skill_slider = mo.ui.slider(0, 9, 1, label='Attacker skill', show_value=True)
    def_skill_slider = mo.ui.slider(0, 9, 1, label='Defender skill', show_value=True)
    mo.vstack([att_skill_slider, def_skill_slider])
    return att_skill_slider, def_skill_slider


@app.cell
def _(HitType, mo):
    options = [item.name for item in HitType] + ['Any hit', 'Strong hit', 'Special effect']
    hit_type_dd = mo.ui.dropdown(options, value=HitType.MISS.name)
    hit_type_dd
    return hit_type_dd, options


@app.cell
def _(hit_type_dd):
    result_type = hit_type_dd.value
    return (result_type,)


@app.cell
def _(
    HitType,
    at_least_one_se,
    attacker_skill,
    compute_hit_matrix_hr,
    defender_skill,
    flat_att_skill,
    flat_def_skill,
    hit_pct,
    mo,
    np,
    plt,
    result_type,
    sns,
):
    outcomes = compute_hit_matrix_hr(attacker_skill, defender_skill)

    match result_type:
        case HitType.MISS.name:
            data = outcomes.p_miss
            raw_data = 100 - hit_pct
        
        case HitType.MISS_WITH_DEF_SE.name:
            data = outcomes.p_miss_with_se
            raw_data = at_least_one_se.T * 100
        
        case HitType.WEAK_HIT.name:
            data = outcomes.p_weak_hit
            raw_data = np.zeros_like(data)
        
        case HitType.STRONG_HIT.name:
            data = outcomes.p_strong_hit
            raw_data = hit_pct
        
        case HitType.STRONG_HIT_WITH_SE.name:
            data = outcomes.p_strong_hit_with_se
            raw_data = np.zeros_like(data)

        case HitType.CRITICAL_HIT_WITH_SE.name:
            data = outcomes.p_crit_with_se
            raw_data = np.zeros_like(data)

        case 'Any hit':
            data = outcomes.p_any_hit
            raw_data = hit_pct

        case 'Strong hit': 
            data = outcomes.p_any_strong_hit
            raw_data = hit_pct

        case 'Special effect': 
            data = outcomes.p_attacker_se
            raw_data = at_least_one_se * 100
    

    pct_data = data * 100

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    hr_plot = sns.heatmap(
        data=pct_data, annot=True, xticklabels=flat_att_skill, yticklabels=flat_def_skill
    )

    # clean plot area and avoid double heatmap bars
    plt.figure()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    raw_plot = sns.heatmap(
        data=raw_data, annot=True, xticklabels=flat_att_skill, yticklabels=flat_def_skill
    )

    hr_plot.set(
        title=f"Homebrew - {result_type} %",
        xlabel="Defense Skill",
        ylabel="Attack Skill",
    )

    raw_plot.set(
        title=f"RAW - {result_type} %",
        xlabel="Defense Skill",
        ylabel="Attack Skill",
    )

    mo.hstack([hr_plot, raw_plot])
    return data, hr_plot, outcomes, pct_data, raw_data, raw_plot


if __name__ == "__main__":
    app.run()
