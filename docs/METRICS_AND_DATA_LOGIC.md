# Metrics And Data Logic

## Join Keys

- Base play join: `play_by_play_data.csv` + `epa.csv` on `gameId` + `playId`
  - PBP columns `game_id`, `play_id` are renamed to `gameId`, `playId`
- Player on-field mapping: `participation.csv` melted from `P1..P15` to long form (`player_id`)
- Personnel join: `pp_data.csv` `play id` mapped to `playId` and joined on `playId`

## Team/Role Normalization

- Team strings are normalized to `strip().upper()`
- Role strings are normalized to `strip().lower()`
- This avoids duplicate team choices caused by spacing/case differences.

## Week Mapping

Display week is computed as:
- regular season (`REG`): `week`
- playoffs (`POST`): `week + 18`

So playoff weeks appear as `19-22`.

## Play Eligibility Filter

A play is included only if:
- `pass_play == 1` or `run_play == 1`
- and all of the following are not 1:
  - `no_play`
  - `spiked_ball`
  - `kneel_down`
  - `two_point_att`
  - `kickoff`
  - `punt`
  - `field_goal`
  - `extra_point`

## Derived Fields

- `yards_gained = passing_yards + rushing_yards`
- `display_week` as described above

## Success Rate Definition

For each eligible play:
- 1st down success: `yards_gained >= 0.4 * distance`
- 2nd down success: `yards_gained >= 0.6 * distance`
- 3rd/4th success: `yards_gained >= distance`
- `distance` must be `> 0`

## On/Off Definitions

By selected `team` and `role`:
- offense role uses team plays where `offense == team`
- defense role uses team plays where `defense == team`

On-field match:
- `Any selected player on field`: play is `On` if at least one selected player is present
- `All selected players on field`: play is `On` only if all selected players are present

Off-field is the complement over the same team-play set.

## Score Margin Filter

Score margin is team-perspective:
- offense plays: `offense_score - defense_score`
- defense plays: `defense_score - offense_score`

Selected margin range is applied per role before summary.

## Metrics

For any filtered split:
- `Plays = count`
- `EPA/play = mean(epa)`
- `Success Rate = mean(success_flag)`
- `Pass Rate = mean(pass_play == 1)`
- `Run Rate = mean(run_play == 1)`

## Baselines

Baselines are computed with currently selected seasons/weeks/filters:
- `Team`: selected team subset
- `League`: all plays passing current filters for that role context

## Trends

Trend tables group by:
- `display_week`
- `is_on` (0/1)

Outputs:
- `On EPA/play`, `Off EPA/play`
- `On Plays`, `Off Plays`

## Personnel Splits

Personnel comes from `pp_data.csv` and is summarized by:
- `Personnel` (numeric code, or `Unknown`)
- `Split` (`On`/`Off`)
