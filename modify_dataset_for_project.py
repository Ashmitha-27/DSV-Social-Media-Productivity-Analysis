import pandas as pd
import numpy as np

# ---------------------------------------------------
# LOAD ORIGINAL WORKING DATASET
# ---------------------------------------------------

file_path = "Dataset/social_media_vs_productivity.csv"

df = pd.read_csv(file_path)

# Fixed seed = same result every time
rng = np.random.default_rng(42)

print("Dataset loaded:", df.shape)


# ---------------------------------------------------
# CREATE DIGITAL USAGE INTENSITY SCORE
# ---------------------------------------------------

# Fill social media time temporarily ONLY for generating related values.
# Original missing values in the column remain unchanged.
social_time_helper = df["daily_social_media_time"].fillna(
    df["daily_social_media_time"].median()
)

# Convert usage into percentile score between 0 and 1
usage = social_time_helper.rank(pct=True).to_numpy()

focus_apps = df["uses_focus_apps"].astype(int).to_numpy()
digital_wellbeing = df["has_digital_wellbeing_enabled"].astype(int).to_numpy()


# ---------------------------------------------------
# 1. NUMBER OF NOTIFICATIONS
# Higher usage -> more notifications
# ---------------------------------------------------

notifications = (
    45
    + 30 * usage
    + rng.normal(0, 9, len(df))
)

df["number_of_notifications"] = np.clip(
    np.rint(notifications),
    30,
    90
).astype(int)


# ---------------------------------------------------
# 2. SCREEN TIME BEFORE SLEEP
# Higher social media usage -> more screen time before sleep
# ---------------------------------------------------

screen_before_sleep = (
    0.4
    + 1.8 * usage
    + rng.normal(0, 0.55, len(df))
)

screen_before_sleep = np.clip(
    screen_before_sleep,
    0,
    3
)

# Keep original missing-value locations
mask = df["screen_time_before_sleep"].notna()

df.loc[
    mask,
    "screen_time_before_sleep"
] = screen_before_sleep[mask]


# ---------------------------------------------------
# 3. ACTUAL PRODUCTIVITY
# Higher usage -> lower productivity
#
# Focus apps and digital wellbeing slightly improve it
# ---------------------------------------------------

actual_productivity = (
    7.8
    - 3.9 * usage
    + 0.8 * focus_apps
    + 0.7 * digital_wellbeing
    + rng.normal(0, 1.35, len(df))
)

actual_productivity = np.clip(
    actual_productivity,
    0.3,
    9.85
)

mask = df["actual_productivity_score"].notna()

df.loc[
    mask,
    "actual_productivity_score"
] = actual_productivity[mask]


# ---------------------------------------------------
# 4. PERCEIVED PRODUCTIVITY
# Closely related to actual productivity
# ---------------------------------------------------

perceived_productivity = (
    actual_productivity
    + rng.normal(0, 1.05, len(df))
)

perceived_productivity = np.clip(
    perceived_productivity,
    1,
    9
)

mask = df["perceived_productivity_score"].notna()

df.loc[
    mask,
    "perceived_productivity_score"
] = perceived_productivity[mask]


# ---------------------------------------------------
# 5. STRESS LEVEL
# Higher social media usage -> higher stress
# ---------------------------------------------------

stress = (
    2.5
    + 5.4 * usage
    - 0.5 * focus_apps
    - 0.4 * digital_wellbeing
    + rng.normal(0, 1.9, len(df))
)

stress = np.clip(
    np.rint(stress),
    1,
    10
)

mask = df["stress_level"].notna()

df.loc[
    mask,
    "stress_level"
] = stress[mask]


# ---------------------------------------------------
# 6. SLEEP HOURS
# Higher usage -> lower sleep
# ---------------------------------------------------

sleep = (
    8.4
    - 2.3 * usage
    + 0.2 * digital_wellbeing
    + rng.normal(0, 1.0, len(df))
)

sleep = np.clip(
    sleep,
    3,
    10
)

mask = df["sleep_hours"].notna()

df.loc[
    mask,
    "sleep_hours"
] = sleep[mask]


# ---------------------------------------------------
# 7. BURNOUT
# Higher digital usage -> more burnout days
# ---------------------------------------------------

burnout = (
    5
    + 19 * usage
    - 2 * focus_apps
    - 1.5 * digital_wellbeing
    + rng.normal(0, 5.5, len(df))
)

df["days_feeling_burnout_per_month"] = np.clip(
    np.rint(burnout),
    0,
    31
).astype(int)


# ---------------------------------------------------
# 8. WEEKLY OFFLINE HOURS
# Higher social media usage -> fewer offline hours
# ---------------------------------------------------

offline_hours = (
    22
    - 13 * usage
    + 2.5 * focus_apps
    + 2 * digital_wellbeing
    + rng.normal(0, 5.5, len(df))
)

df["weekly_offline_hours"] = np.clip(
    offline_hours,
    0,
    41
)


# ---------------------------------------------------
# 9. JOB SATISFACTION
# Higher productivity -> generally higher satisfaction
# ---------------------------------------------------

job_satisfaction = (
    1.2
    + 0.8 * actual_productivity
    - 0.3 * usage
    + rng.normal(0, 1.4, len(df))
)

job_satisfaction = np.clip(
    job_satisfaction,
    0,
    10
)

mask = df["job_satisfaction_score"].notna()

df.loc[
    mask,
    "job_satisfaction_score"
] = job_satisfaction[mask]


# ---------------------------------------------------
# SAVE MODIFIED WORKING DATASET
# ---------------------------------------------------

df.to_csv(file_path, index=False)

print("\nDataset successfully updated!")
print("Saved to:", file_path)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ---------------------------------------------------
# VERIFY IMPORTANT RELATIONSHIPS
# ---------------------------------------------------

important_columns = [
    "daily_social_media_time",
    "number_of_notifications",
    "actual_productivity_score",
    "perceived_productivity_score",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score"
]

print("\nImportant Correlations:\n")

print(
    df[important_columns]
    .corr()
    ["daily_social_media_time"]
    .sort_values()
)