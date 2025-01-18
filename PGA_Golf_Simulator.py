# from rich.table import Table
from scipy.stats import truncnorm
from rich.console import Console
import pandas as pd
import random
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QHBoxLayout, QSpacerItem, QSizePolicy, QWidget
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize a Console object
console = Console()

# Load each CSV file into a dataFrame

#The percentage of times a player makes par or better after missing the green in regulation.
scrambling_df = pd.read_csv('scrambling.csv')

# The average number of strokes a player takes per round.
scoring_average_df = pd.read_csv('scoring_average.csv')

# Assesses the number of strokes a player gains or loses on the green compared to the field average.
putting_df = pd.read_csv('putting.csv')

# The proportion of holes where a player reaches the green in the expected number of strokes (par minus two).
greens_in_regulation_df = pd.read_csv('greens_in_regulation.csv')

# The average distance a player hits the ball off the tee on measured drives.
driving_distance_df = pd.read_csv('driving_distance.csv')

# The average number of birdies a player makes per round.
birdie_average_df = pd.read_csv('birdie_average.csv')

# The percentage of tee shots that land in the fairway.
driving_accuracy_df = pd.read_csv('driving_accuracy.csv')

# Measures the number of strokes a player gains or loses on approach shots compared to the field average.
approach_the_green_df = pd.read_csv('approach_the_green.csv')

# Measures a player's performance on tee shots on par 4s and par 5s relative to the field.
off_the_tee_df = pd.read_csv('off_the_tee.csv')

# The overall number of strokes a player gains or loses relative to the field across all aspects of play.
strokes_gained_df = pd.read_csv('strokes_gained.csv')

# Combines strokes gained from driving, approach shots, and around-the-green play, excluding putting.
tee_to_green_df = pd.read_csv('tee_to_green.csv')

#Evaluates a player's performance on shots within 30 yards of the green, excluding putting, relative to the field.
around_the_green_df = pd.read_csv('around_the_green.csv')

# The average number of putts taken per green in regulation.
putting_average_df = pd.read_csv('putting_average.csv')

# The percentage of holes where a player avoids taking three or more putts.
thirdputt_avoid_df = pd.read_csv('thirdputt_avoid.csv')

# The percentage of times a player scores a birdie or better when hitting approach shots from the rough.
layup_df = pd.read_csv('layup.csv')

# Adjust the way that each value shows to merge them into a sole dataset
scrambling_df['SCR_%'] = scrambling_df['SCR_%'].str.replace('%', '')
scrambling_df['SCR_%'] = scrambling_df['SCR_%'].astype(float) / 100
greens_in_regulation_df['GREENS_%'] = greens_in_regulation_df['GREENS_%'].str.replace('%', '')
greens_in_regulation_df['GREENS_%'] = greens_in_regulation_df['GREENS_%'].astype(float) / 100
driving_accuracy_df['DRIVE_ACC%'] = driving_accuracy_df['DRIVE_ACC%'].str.replace('%', '')
driving_accuracy_df['DRIVE_ACC%'] = driving_accuracy_df['DRIVE_ACC%'].astype(float) / 100
thirdputt_avoid_df["3P_AVOID_%"] = thirdputt_avoid_df["3P_AVOID_%"].str.replace('%', '')
thirdputt_avoid_df["3P_AVOID_%"] = thirdputt_avoid_df["3P_AVOID_%"].astype(float) / 100
layup_df["LAYUP_%"] = layup_df["LAYUP_%"].str.replace('%', '')
layup_df['LAYUP_%'] = layup_df['LAYUP_%'].astype(float) / 100

# Merge all the CSV files
merged_df = scoring_average_df.merge(scrambling_df, on='PLAYER') \
                          .merge(putting_df, on='PLAYER') \
                          .merge(greens_in_regulation_df, on='PLAYER') \
                          .merge(driving_distance_df, on='PLAYER') \
                          .merge(birdie_average_df, on='PLAYER') \
                          .merge(driving_accuracy_df, on='PLAYER') \
                          .merge(approach_the_green_df, on='PLAYER') \
                          .merge(off_the_tee_df, on='PLAYER') \
                          .merge(strokes_gained_df, on='PLAYER') \
                          .merge(tee_to_green_df, on='PLAYER') \
                          .merge(around_the_green_df, on='PLAYER') \
                          .merge(putting_average_df, on='PLAYER') \
                          .merge(thirdputt_avoid_df, on='PLAYER') \
                          .merge(layup_df, on='PLAYER') \
# print(merged_df)
# Show everything
print(merged_df.to_string(index=True))

# Convert a merged dataframe into a list of dictionary
players = merged_df.to_dict(orient='records')

# Define a class for a hole
class Hole:
    def __init__(self, hole_number, par, yardage):
        self.hole_number = hole_number
        self.par = par
        self.yardage = yardage

# Create a list of holes for Augusta National
augusta_holes = [
    Hole(1, 4, 445),
    Hole(2, 5, 575),
    Hole(3, 4, 360),
    Hole(4, 3, 240),
    Hole(5, 4, 455),
    Hole(6, 3, 180),
    Hole(7, 4, 450),
    Hole(8, 5, 570),
    Hole(9, 4, 460),
    Hole(10, 4, 420),
    Hole(11, 4, 520),
    Hole(12, 3, 155),
    Hole(13, 5, 510),
    Hole(14, 4, 440),
    Hole(15, 5, 550),
    Hole(16, 3, 170),
    Hole(17, 4, 440),
    Hole(18, 4, 465)
]

# Simulate a golf tournament round for a single player.
class pgaSimulation:
    def __init__(self, player):
        # Initiate the player and default value for the simulation
        self.player = player
        self.total_strokes = 0
        self.hole_strokes = 0
        self.distance_remaining = 0
        self.fairway_rough_green = None

        # Initiate the player's ability based upon the stats the player has
        self.driver_accuracy = player["DRIVE_ACC%"]
        self.scrambling = player["SCR_%"]
        self.putting_strokes = player["SG_Putt"]
        self.greens_in_regulation = player["GREENS_%"]
        self.birdie_average = player["BIRDIE_AVG"]
        self.driver_distance = player["DRIVE_AVG"]
        self.scoring_average = player["SCORE_AVG"]
        self.approach_the_green = player["SG_Approach"]
        self.off_the_tee = player["SG_OffTheTee"]
        self.strokes_gained = player["SG_All"]
        self.tee_to_green = player["SG_TeeToGreen"]
        self.around_the_green = player["SG_AroundTheGreen"]
        self.putting_average = player["PUTT_AVG"]
        self.thirdputt_avoid = player["3P_AVOID_%"]
        self.layup = player["LAYUP_%"]

        # Setting up the clubs that the player will play with
        # The value of distance for each club is counted backwards from their driver_distance
        # Putter has a flat value since there should be no advantage of how far they can putt
        self.clubs = {
            "Driver": int(self.driver_distance),
            "3 Wood": int(self.driver_distance - 30),
            "5 Wood": int(self.driver_distance - 60),
            "4 Iron": int(self.driver_distance - 75),
            "5 Iron": int(self.driver_distance - 90),
            "6 Iron": int(self.driver_distance - 105),
            "7 Iron": int(self.driver_distance - 120),
            "8 Iron": int(self.driver_distance - 135),
            "9 Iron": int(self.driver_distance - 150),
            "Pitching Wedge": int(self.driver_distance - 165),
            "50 Degree": int(self.driver_distance - 190),
            "54 Degree": int(self.driver_distance - 205),
            "58 Degree": int(self.driver_distance - 220),
            "Putter": 20,
        }

        # Setting up binary outcomes (basically success or failure) for all the situations that they would come across
        # and the weights of success for the sake of alignment
        self.putting_outcomes = ["in the hole", "green"]
        self.putting_weights = [1 - self.thirdputt_avoid + 0.1, self.thirdputt_avoid - 0.1]
        self.driver_outcomes = ["fairway", "rough"]
        self.driver_weights = [self.driver_accuracy, 1 - self.driver_accuracy]
        self.fairway_to_green_outcomes = ["green", "rough_around_green"]
        self.fairway_to_green_weights = [self.greens_in_regulation, 1 - self.greens_in_regulation]
        self.sc_rough_to_green_outcomes = ["green", "rough_around_green"]
        self.sc_rough_to_green_weights = [self.scrambling, 1 - self.scrambling]
        self.rough_to_green_outcomes = ["green", "rough_around_green"]
        self.rough_to_green_weights = [self.scrambling, 1 - self.scrambling]
        self.layup_from_fairway_outcomes = ["fairway", "rough"]
        self.layup_from_fairway_weights = [0.7 + self.layup, 1 - self.layup - 0.7]
        self.layup_from_rough_outcomes = ["fairway", "rough"]
        self.layup_from_rough_weights = [0.6 + self.layup, 1 - self.layup - 0.6]

    def first_shot(self, hole):
        # Execute driver shot to land the ball to the fairway/rough
        # The tee-off with driver on par 4/par 5
        if hole.par != 3:
            # Uniform distribution works for the driver distance because of its consistency
            # Determine the outcome based on the binary consequences and the alignment that were set up above
            drive_distance = random.randint(self.clubs["Driver"] - 10, self.clubs["Driver"] + 10)
            self.fairway_rough_green = random.choices(self.driver_outcomes, weights=self.driver_weights, k=1)[0]

            self.hole_strokes += 1
            # Make the remaining distance updated
            self.distance_remaining -= drive_distance

        # On per 3
        else:
            # Select an appropriate club for the distance to the hole
            # Check the appropriate club from the bottom (putter)
            # so that the distance with the selected club can be the closest to the distance on par 3
            # The reason for "reversed" being the value should be MORE THAN or equal to the distance
            for club, distance in reversed(self.clubs.items()):
                if distance >= self.distance_remaining:
                    # Set desired mean to the target peak (10 feet)
                    desired_mean = 10
                    # Set sigma to create the distribution
                    sigma = 0.5
                    adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (
                            # Set up large weights associated with the shot to the green
                            (1 - self.greens_in_regulation) * 1.5 +
                            (1 - self.approach_the_green) * 2 +
                            # Set up small weights associated with every aspect
                            (1 - self.off_the_tee) * 0.2 +
                            (1 - self.strokes_gained) * 0.2 +
                            (1 - self.tee_to_green) * 0.2 +
                            (1 - self.birdie_average) * 0.2 +
                            (1 - self.scoring_average) * 0.2
                    )

                    # Calculate shot distance and update remaining distance
                    shot_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - shot_distance)
                    # Limit the remaining distance to a maximum of 150 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 150)

                    # Let the ball land on the green if the remaining distance is less than or equal to 20 (feet)
                    if 0 < self.distance_remaining <= 20:
                        self.fairway_rough_green = "green"
                    # Make a random choice of "green" or "rough" based upon the weights
                    # if the distance is between 20 and 40
                    elif 20 < self.distance_remaining <= 40:
                        self.fairway_rough_green = \
                        random.choices(self.fairway_to_green_outcomes, weights=self.fairway_to_green_weights, k=1)[0]
                    # Let the ball land on the rough if the remaining distance is more than 40
                    else:
                        self.fairway_rough_green = "rough_around_green"

                    self.hole_strokes += 1
                    break

    def second_shot(self, hole):
        # From the fairway
        if self.hole_strokes == 1 and self.fairway_rough_green == "fairway":
            # Laying up on par 5 (if the player can't hit the ball to the green with any clubs except driver)
            if hole.par == 5 and self.distance_remaining > self.clubs["3 Wood"]:
                # Choose an appropriate club
                for club, distance in reversed(self.clubs.items()):
                    # Keep iterating through clubs to the point where the remaining distance will be less than 100 yards
                    # in which case the ball will be always around 100 yards away from the hole
                    if self.distance_remaining - distance <= 100:
                        adjustment = np.random.lognormal(2, 1) * (
                                # Set up large weights associated with layup
                                (1 - self.layup) * 0.5 +
                                # Set up small weights associated with every aspect
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )
                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = 100 + abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 150 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                        self.fairway_rough_green = random.choices(self.layup_from_fairway_outcomes, weights=self.layup_from_fairway_weights, k=1)[0]
                        self.hole_strokes += 1

                        break
            # To the green on par 5
            elif hole.par == 5 and self.distance_remaining <= self.clubs["3 Wood"]:
                for club, distance in reversed(self.clubs.items()):
                    # Select the club that has the closest value to the distance to the hole
                    if distance >= self.distance_remaining:
                        desired_mean = 15
                        sigma = 0.6
                        # Calculate dynamic variance
                        adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (
                                # Set up large weights associated with the shot to the green
                                (1 - self.greens_in_regulation) * 1.5 +
                                (1 - self.approach_the_green) * 2 +
                                # Set up small weights associated with every aspect
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 300 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 300)
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        elif 20 < self.distance_remaining <= 40:
                            self.fairway_rough_green = random.choices(self.fairway_to_green_outcomes, weights=self.fairway_to_green_weights, k=1)[0]
                        else:
                            self.fairway_rough_green = "rough_around_green"
                        self.hole_strokes += 1

                        break
            # To the green on par 4
            elif hole.par == 4:
                # The same process as "To the green on par 5"
                for club, distance in reversed(self.clubs.items()):
                    if distance >= self.distance_remaining:
                        desired_mean = 10
                        sigma = 0.5
                        adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (
                                # Set up large weights associated with the shot to the green
                                (1 - self.greens_in_regulation) * 1.5 +
                                (1 - self.approach_the_green) * 2 +
                                # Set up small weights associated with every aspect
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 150 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        elif 20 < self.distance_remaining <= 40:
                            self.fairway_rough_green = random.choices(self.fairway_to_green_outcomes, weights=self.fairway_to_green_weights, k=1)[0]
                        else:
                            self.fairway_rough_green = "rough_around_green"
                        self.hole_strokes += 1

                        break

        # Second shot from the rough on par 4 or par 5
        if self.hole_strokes == 1 and self.fairway_rough_green == "rough":
            # Laying up on par 5
            # Remove the options of using driver and woods because
            # it would not be realistic to shot with driver/woods from the rough
            if hole.par == 5 and self.distance_remaining > self.clubs["4 Iron"]:
                for club, distance in reversed(self.clubs.items()):
                    if club == "Driver":
                        continue
                    # The remaining distance set as 150, which is 50 yards further than the one on the fairway
                    # It implies the fact that players tend to have short clubs on the rough for safety
                    # The range of random values is a bit wider due to the rough situation
                    elif self.distance_remaining - distance <= 150:
                        # Determine the outcome (shot distance)
                        adjustment = np.random.lognormal(2, 1) * (
                                # Set up large weights associated with the shot from the rough
                                (1 - self.scrambling) * 1.5 +
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )
                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = 100 + abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 150 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                        self.fairway_rough_green = random.choices(self.layup_from_rough_outcomes, weights=self.layup_from_rough_weights, k=1)[0]
                        self.distance_remaining -= shot_distance
                        self.hole_strokes += 1

                        break
            # To the green on par 5
            elif hole.par == 5 and self.distance_remaining <= self.clubs["4 Iron"]:
                for club, distance in reversed(self.clubs.items()):
                    if club == "Driver":
                        continue
                    elif distance >= self.distance_remaining:
                        desired_mean = 20
                        sigma = 0.6
                        adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (
                                # Set up medium weights associated with the shot to the green
                                (1 - self.greens_in_regulation) +
                                (1 - self.approach_the_green) +
                                # Set up large weights associated with the shot from the rough
                                (1 - self.scrambling) * 7 +

                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 300 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 300)
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        elif 20 < self.distance_remaining <= 40:
                            self.fairway_rough_green = random.choices(self.rough_to_green_outcomes, weights=self.rough_to_green_weights, k=1)[0]
                        else:
                            self.fairway_rough_green = "rough_around_green"
                        self.hole_strokes += 1

                        break
            # To the green on par 4
            elif hole.par == 4:
                for club, distance in reversed(self.clubs.items()):
                    if distance >= self.distance_remaining:
                        desired_mean = 15
                        sigma = 0.5
                        adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (
                                # Set up medium weights associated with the shot to the green
                                (1 - self.greens_in_regulation) +
                                (1 - self.approach_the_green) +
                                # Set up large weights associated with the shot from the rough
                                (1 - self.scrambling) * 7 +
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 150 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        elif 20 < self.distance_remaining <= 40:
                            self.fairway_rough_green = random.choices(self.rough_to_green_outcomes, weights=self.rough_to_green_weights, k=1)[0]
                        else:
                            self.fairway_rough_green = "rough_around_green"
                        self.hole_strokes += 1

                        break
            # To the green on par 3 (Approach)
            elif hole.par == 3 and self.fairway_rough_green == "rough_around_green":
                for club, distance in reversed(self.clubs.items()):
                    # The range of random values is comparatively small
                    # because this shot is expected to be executed from the rough around the green
                    if distance >= self.distance_remaining:
                        # Calculate dynamic variance
                        dynamic_variance = max(1, 1 - (self.scoring_average * 0.05))
                        # Center the distribution around 2 feet
                        mean_bias = 2
                        std_dev = dynamic_variance
                        # Truncate between 0 and 15
                        a, b = (0 - mean_bias) / std_dev, (15 - mean_bias) / std_dev

                        # Calculate adjustment
                        adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (
                                # Set up medium weights associated with the shot from the rough
                                (1 - self.scrambling) * 1.5 +
                                # Set up large weights associated with approach shots
                                (1 - self.around_the_green) * 5 +

                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 40 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 40)
                        # The shot with the marge of less than or equal to 20 feet is treated as "on the green"
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        # Make a random choice for ANY distance more than 20
                        # It may not be realistic to hit the ball much far away from the hole with approach shot
                        else:
                            self.fairway_rough_green = random.choices(self.sc_rough_to_green_outcomes, weights=self.sc_rough_to_green_weights, k=1)[0]
                        self.hole_strokes += 1

                        break
            # To the hole on par 3 (Putting)
            elif hole.par == 3 and self.fairway_rough_green == "green":

                # Divide putting situations by specific remaining distance (if self.distance_remaining >= x:)
                # reflects the fact that the closer the ball is to the hole, the easier the putting is
                # Therefore, the range of random value is wider as the remaining distance gets further

                if self.distance_remaining >= 12:
                    mean_bias = 3 + ((1 - self.putting_average) * 0.8)
                    std_dev = 2 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (
                            # Set up large weights associated with putting
                            (1 - self.putting_strokes) +
                            # Set up small weights associated with approach
                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        # Make a random choice for the remaining distance between 1 foot and 4 feet
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    # If the distance to the hole is less than 1 foot
                    # "in the hole" conceded
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance


                elif self.distance_remaining >= 7:
                    # Calculate dynamic variance
                    mean_bias = 2 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1.5 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance

                elif self.distance_remaining >= 4:
                    # Calculate dynamic variance
                    mean_bias = 1 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1.5 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance

                elif self.distance_remaining >= 1:
                    # Calculate dynamic variance
                    mean_bias = 0.5 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    else:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                else:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0


    def third_shot(self):
        # On par 5 (after laying up)
        # From the fairway
        # The algorithm itself is pretty much the same as the second shot from the fairway to the green
        if self.hole_strokes == 2 and self.fairway_rough_green == "fairway":
            for club, distance in reversed(self.clubs.items()):
                if distance >= self.distance_remaining:
                    desired_mean = 10
                    sigma = 0.5
                    # Calculate dynamic variance
                    adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (

                            (1 - self.greens_in_regulation) * 1.5 +
                            (1 - self.approach_the_green) * 2 +

                            (1 - self.off_the_tee) * 0.2 +
                            (1 - self.strokes_gained) * 0.2 +
                            (1 - self.tee_to_green) * 0.2 +
                            (1 - self.birdie_average) * 0.2 +
                            (1 - self.scoring_average) * 0.2
                    )

                    # Calculate shot distance and update remaining distance
                    shot_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - shot_distance)
                    # Limit the remaining distance to a maximum of 150 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                    if 0 < self.distance_remaining <= 20:
                        self.fairway_rough_green = "green"
                    elif 20 < self.distance_remaining <= 40:
                        self.fairway_rough_green = random.choices(self.fairway_to_green_outcomes, weights=self.fairway_to_green_weights, k=1)[0]
                    else:
                        self.fairway_rough_green = "rough_around_green"
                    self.hole_strokes += 1
                    break

        # From the rough
        # The algorithm is pretty much the same as the second shot from the rough to the green
        elif self.hole_strokes == 2 and self.fairway_rough_green == "rough":
            for club, distance in reversed(self.clubs.items()):
                if distance >= self.distance_remaining:
                    desired_mean = 15
                    sigma = 0.5
                    # Calculate dynamic variance
                    adjustment = np.random.lognormal(mean=np.log(desired_mean), sigma=sigma) * (

                            (1 - self.greens_in_regulation) +
                            (1 - self.approach_the_green) +

                            (1 - self.scrambling) * 7 +

                            (1 - self.off_the_tee) * 0.2 +
                            (1 - self.strokes_gained) * 0.2 +
                            (1 - self.tee_to_green) * 0.2 +
                            (1 - self.birdie_average) * 0.2 +
                            (1 - self.scoring_average) * 0.2
                    )

                    # Calculate shot distance and update remaining distance
                    shot_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - shot_distance)
                    # Limit the remaining distance to a maximum of 150 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 150)
                    if 0 < self.distance_remaining <= 20:
                        self.fairway_rough_green = "green"
                    elif 7 < self.distance_remaining <= 40:
                        self.fairway_rough_green = random.choices(self.rough_to_green_outcomes, weights=self.rough_to_green_weights, k=1)[0]
                    else:
                        self.fairway_rough_green = "rough"
                    self.hole_strokes += 1
                    break
        # Approach
        elif self.hole_strokes == 2 and self.fairway_rough_green == "rough_around_green":
            for club, distance in reversed(self.clubs.items()):
                if distance >= self.distance_remaining:
                    # Calculate dynamic variance
                    dynamic_variance = max(1, 1 - (self.scoring_average * 0.05))
                    # Center the distribution around 5 feet
                    mean_bias = 2
                    std_dev = dynamic_variance
                    # Truncate between 0 and 10
                    a, b = (0 - mean_bias) / std_dev, (15 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.scrambling) * 1.5 +

                            (1 - self.around_the_green) * 5 +

                            (1 - self.off_the_tee) * 0.2 +
                            (1 - self.strokes_gained) * 0.2 +
                            (1 - self.tee_to_green) * 0.2 +
                            (1 - self.birdie_average) * 0.2 +
                            (1 - self.scoring_average) * 0.2
                    )

                    # Calculate shot distance and update remaining distance
                    shot_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - shot_distance)
                    # Limit the remaining distance to a maximum of 40 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 40)
                    # The shot with the marge of less than or equal to 20 yards is treated as "on the green"
                    if 0 < self.distance_remaining <= 20:
                        self.fairway_rough_green = "green"
                    # Make a random choice for ANY distance more than 20
                    # It may not be realistic to hit the ball much far away from the hole with approach shot
                    else:
                        self.fairway_rough_green = \
                        random.choices(self.sc_rough_to_green_outcomes, weights=self.sc_rough_to_green_weights, k=1)[0]
                    self.hole_strokes += 1

                    break

        # The putt for birdie on Par 4, or the putt for par on par 3
        # The pattern is the same as the putting at the second shot on par 3
        elif self.hole_strokes == 2 and self.fairway_rough_green == "green":
            if self.distance_remaining >= 12:
                mean_bias = 3 + ((1 - self.putting_average) * 0.8)
                std_dev = 2 + (1 - self.putting_average) * 0.3
                a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                # Calculate adjustment
                adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (
                    # Set up large weights associated with putting
                        (1 - self.putting_strokes) +
                        # Set up small weights associated with approach
                        (1 - self.around_the_green) * 0.1 +

                        (1 - self.off_the_tee) * 0.1 +
                        (1 - self.strokes_gained) * 0.1 +
                        (1 - self.tee_to_green) * 0.1 +
                        (1 - self.birdie_average) * 0.1 +
                        (1 - self.scoring_average) * 0.1
                )

                # Calculate shot distance and update remaining distance
                putt_distance = self.distance_remaining + adjustment
                self.distance_remaining = abs(self.distance_remaining - putt_distance)
                # Limit the remaining distance to a maximum of 7 feet
                self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                if self.distance_remaining <= 1:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0
                elif 1 < self.distance_remaining <= 4:
                    # Make a random choice for the remaining distance between 1 foot and 4 feet
                    hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                    if hole_green == "in the hole":
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif hole_green == "green":
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance
                # If the distance to the hole is less than 1 foot
                # "in the hole" conceded
                else:
                    self.hole_strokes += 1
                    # Move onto the next shot
                    self.distance_remaining -= putt_distance


            elif self.distance_remaining >= 7:
                # Calculate dynamic variance
                mean_bias = 2 + ((1 - self.putting_average) * 0.8)
                std_dev = 1.5 + (1 - self.putting_average) * 0.3
                a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                # Calculate adjustment
                adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                        (1 - self.putting_strokes) +

                        (1 - self.around_the_green) * 0.1 +

                        (1 - self.off_the_tee) * 0.1 +
                        (1 - self.strokes_gained) * 0.1 +
                        (1 - self.tee_to_green) * 0.1 +
                        (1 - self.birdie_average) * 0.1 +
                        (1 - self.scoring_average) * 0.1
                )

                # Calculate shot distance and update remaining distance
                putt_distance = self.distance_remaining + adjustment
                self.distance_remaining = abs(self.distance_remaining - putt_distance)
                # Limit the remaining distance to a maximum of 7 feet
                self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                if self.distance_remaining <= 1:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0
                elif 1 < self.distance_remaining <= 4:
                    hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                    if hole_green == "in the hole":
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif hole_green == "green":
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance
                else:
                    self.hole_strokes += 1
                    # Move onto the next shot
                    self.distance_remaining -= putt_distance

            elif self.distance_remaining >= 4:
                # Calculate dynamic variance
                mean_bias = 1 + ((1 - self.putting_average) * 0.8)
                std_dev = 1.5 + (1 - self.putting_average) * 0.3
                a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                # Calculate adjustment
                adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                        (1 - self.putting_strokes) +

                        (1 - self.around_the_green) * 0.1 +

                        (1 - self.off_the_tee) * 0.1 +
                        (1 - self.strokes_gained) * 0.1 +
                        (1 - self.tee_to_green) * 0.1 +
                        (1 - self.birdie_average) * 0.1 +
                        (1 - self.scoring_average) * 0.1
                )

                # Calculate shot distance and update remaining distance
                putt_distance = self.distance_remaining + adjustment
                self.distance_remaining = abs(self.distance_remaining - putt_distance)
                # Limit the remaining distance to a maximum of 7 feet
                self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                if self.distance_remaining <= 1:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0
                elif 1 < self.distance_remaining <= 4:
                    hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                    if hole_green == "in the hole":
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif hole_green == "green":
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance
                else:
                    self.hole_strokes += 1
                    # Move onto the next shot
                    self.distance_remaining -= putt_distance

            elif self.distance_remaining >= 1:
                # Calculate dynamic variance
                mean_bias = 0.5 + ((1 - self.putting_average) * 0.8)
                std_dev = 1 + (1 - self.putting_average) * 0.3
                a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                # Calculate adjustment
                adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                        (1 - self.putting_strokes) +

                        (1 - self.around_the_green) * 0.1 +

                        (1 - self.off_the_tee) * 0.1 +
                        (1 - self.strokes_gained) * 0.1 +
                        (1 - self.tee_to_green) * 0.1 +
                        (1 - self.birdie_average) * 0.1 +
                        (1 - self.scoring_average) * 0.1
                )

                # Calculate shot distance and update remaining distance
                putt_distance = self.distance_remaining + adjustment
                self.distance_remaining = abs(self.distance_remaining - putt_distance)
                # Limit the remaining distance to a maximum of 7 feet
                self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                if self.distance_remaining <= 1:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0
                else:
                    hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                    if hole_green == "in the hole":
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif hole_green == "green":
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance
            else:
                self.hole_strokes += 1
                # Make sure to make the remaining distance 0 to move onto the next hole
                self.distance_remaining = 0

    def shot_till_InTheHole(self):
        # Keep executing shots from the rough until the ball lands on the green
        # This means the player would go in a circle within this function after their 4th shot
        # unless they hit the ball on the green
        if self.hole_strokes >= 3 and self.fairway_rough_green == "rough":
            while self.fairway_rough_green != "green":
                for club, distance in reversed(self.clubs.items()):
                    if distance >= self.distance_remaining:
                        # Calculate dynamic variance
                        dynamic_variance = max(1, 1 - (self.scoring_average * 0.05))
                        # Center the distribution around 2 feet
                        mean_bias = 2
                        std_dev = dynamic_variance
                        # Truncate between 0 and 15
                        a, b = (0 - mean_bias) / std_dev, (15 - mean_bias) / std_dev

                        # Calculate adjustment
                        adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (
                                (1 - self.scrambling) * 1.5 +
                                (1 - self.around_the_green) * 5 +
                                (1 - self.off_the_tee) * 0.2 +
                                (1 - self.strokes_gained) * 0.2 +
                                (1 - self.tee_to_green) * 0.2 +
                                (1 - self.birdie_average) * 0.2 +
                                (1 - self.scoring_average) * 0.2
                        )

                        # Calculate shot distance and update remaining distance
                        shot_distance = self.distance_remaining + adjustment
                        self.distance_remaining = abs(self.distance_remaining - shot_distance)
                        # Limit the remaining distance to a maximum of 40 feet
                        self.distance_remaining = np.clip(self.distance_remaining, None, 40)
                        # The shot with the marge of less than or equal to 20 yards is treated as "on the green"
                        if 0 < self.distance_remaining <= 20:
                            self.fairway_rough_green = "green"
                        else:
                            self.fairway_rough_green = \
                                random.choices(self.sc_rough_to_green_outcomes, weights=self.sc_rough_to_green_weights,
                                               k=1)[0]
                        self.hole_strokes += 1

                        break
        # Keep putting on the green until the ball gets in the hole
        # This means the player would go in a circle within this function after their 4th shot
        # unless they made the putt
        if self.hole_strokes >= 3 and self.fairway_rough_green == "green":
            while self.distance_remaining != 0:
                if self.distance_remaining >= 12:
                    mean_bias = 3 + ((1 - self.putting_average) * 0.8)
                    std_dev = 2 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (
                        # Set up large weights associated with putting
                            (1 - self.putting_strokes) +
                            # Set up small weights associated with approach
                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        # Make a random choice for the remaining distance between 1 foot and 4 feet
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    # If the distance to the hole is less than 1 foot
                    # "in the hole" conceded
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance


                elif self.distance_remaining >= 7:
                    # Calculate dynamic variance
                    mean_bias = 2 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1.5 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance

                elif self.distance_remaining >= 4:
                    # Calculate dynamic variance
                    mean_bias = 1 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1.5 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    elif 1 < self.distance_remaining <= 4:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                    else:
                        self.hole_strokes += 1
                        # Move onto the next shot
                        self.distance_remaining -= putt_distance

                elif self.distance_remaining >= 1:
                    # Calculate dynamic variance
                    mean_bias = 0.5 + ((1 - self.putting_average) * 0.8)
                    std_dev = 1 + (1 - self.putting_average) * 0.3
                    a, b = (0 - mean_bias) / std_dev, (4 - mean_bias) / std_dev

                    # Calculate adjustment
                    adjustment = truncnorm.rvs(a, b, loc=mean_bias, scale=std_dev) * (

                            (1 - self.putting_strokes) +

                            (1 - self.around_the_green) * 0.1 +

                            (1 - self.off_the_tee) * 0.1 +
                            (1 - self.strokes_gained) * 0.1 +
                            (1 - self.tee_to_green) * 0.1 +
                            (1 - self.birdie_average) * 0.1 +
                            (1 - self.scoring_average) * 0.1
                    )

                    # Calculate shot distance and update remaining distance
                    putt_distance = self.distance_remaining + adjustment
                    self.distance_remaining = abs(self.distance_remaining - putt_distance)
                    # Limit the remaining distance to a maximum of 7 feet
                    self.distance_remaining = np.clip(self.distance_remaining, None, 7)
                    if self.distance_remaining <= 1:
                        self.hole_strokes += 1
                        # Make sure to make the remaining distance 0 to move onto the next hole
                        self.distance_remaining = 0
                    else:
                        hole_green = random.choices(self.putting_outcomes, weights=self.putting_weights, k=1)[0]
                        if hole_green == "in the hole":
                            self.hole_strokes += 1
                            # Make sure to make the remaining distance 0 to move onto the next hole
                            self.distance_remaining = 0
                        elif hole_green == "green":
                            self.hole_strokes += 1
                            # Move onto the next shot
                            self.distance_remaining -= putt_distance
                else:
                    self.hole_strokes += 1
                    # Make sure to make the remaining distance 0 to move onto the next hole
                    self.distance_remaining = 0

    # The function for properly implementing each function above
    def simulate_round(self, holes):
        # Make sure that the hole-specific value is back to its default
        # once the player moves onto the next hole
        for hole in holes:
            self.distance_remaining = hole.yardage
            self.fairway_rough_green = None
            self.hole_strokes = 0

            self.first_shot(hole)
            self.second_shot(hole)
            self.third_shot()
            self.shot_till_InTheHole()

            # Adding the total stroke for each hole to the total one for the entire round (18 holes)
            self.total_strokes += self.hole_strokes

# Simulate a round for each player
# Function to simulate a single round for all players
# Function to simulate a single round for all players
def simulate_tournament(players, holes):
    results = []
    for player_data in players:
        player_simulation = pgaSimulation(player_data)
        player_simulation.simulate_round(holes)
        results.append((player_data["PLAYER"], player_simulation.total_strokes))
    return results

# Run the simulation 10,000 times with a progress bar
winners = []
for _ in tqdm(range(10000), desc="Simulating tournaments"):
    results = simulate_tournament(players, augusta_holes)
    winner = min(results, key=lambda x: x[1])
    winners.append(winner[0])

# Count the frequency of each winner
winner_counts = Counter(winners)

# Filter players who won more than once
filtered_winners = {player: count for player, count in winner_counts.items() if count > 1}

# Create the chart
plt.figure(figsize=(10, 6))
plt.bar(filtered_winners.keys(), filtered_winners.values())
plt.xlabel('Players')
plt.ylabel('Frequency of Winning')
plt.title('Frequency of Winning for Players Who Won More Than Once')
plt.xticks(rotation=45)
plt.show()

# Save the chart as an image file
plt.savefig('winning_frequency_chart.png')

print("The chart has been saved as 'winning_frequency_chart.png'.")