

![My Photo](/1.png)

# NBA Player Stats Analytics App <img src="icon.png" align="right" />

## Overview


This documentation provides a comprehensive overview of the NBA stats dataset for the years 1998 to 2002. Each attribute has been explained in detail to facilitate a better understanding of the dataset and its potential applications.NBA Stats Dataset Documentation (1997-2023)
Overview
This documentation provides an in-depth explanation of the attributes included in the NBA stats dataset from Kaggle, covering the years 1997 to 2023. The dataset contains information about various player statistics during this time-period.

## Features

- **Yearly Player Stats:** View statistics for NBA players' performance basis for each year.
- **Select Specific Player** View in-depth individual player stats by selecting (click 2 lmb) the player of choice or using filter to find a specific player to view whether by position or year.
- **Data Visualization:** Utilizes scatterplots for visualizing player performance trends and patterns.
- **JSON Data Loading:** The app supports loading player data from a JSON file, allowing users to easily update and customize the dataset.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/nba-analytics-app.git
    ```

2. Install dependencies:

    Pandas

   Matplotlib

   Sk.lean

   Seaborn

   Numpy

   mplcurser

   tkinnter
   
3. Install dependencies:
    ```bash
   pip
    ```
    


   
4. Upload player data:

    - Prepare a csv file with player statistics (e.g., `NBA_Player_Stats.csv`).
  

## Usage

1. Run the app:
    - Run python script in preferred IDE eg.(VSCODE)

2. Load data assets by navigating to the upload CSV button.

3. select a specific data asset (NBA_Player_Stats). then wait for it to load CSV.

4. Explore player statistics and visualizations using the user-friendly interface.



## --To follow instructions better on using the analytic app click the link below--
[Link Text](https://youtu.be/0ammUaaiQ3I)




## Why Scatterplots?

Scatterplots are chosen for their effectiveness in displaying the relationship between two variables. In the context of this project, scatterplots are employed to showcase the correlation between different player statistics, such as points scored, assists, and rebounds per game. This visual representation allows users to identify trends, outliers, and patterns, providing a deeper understanding of player performance over time.


## Why Radarchart?

Radar charts are selected for their efficacy in illustrating the multidimensional aspects of NBA player statistics. In the scope of this project, radar charts are utilized to highlight the correlations among various player performance metrics, including points scored, assists, rebounds per game, and more. This visual depiction enables users to discern trends, outliers, and patterns, fostering a comprehensive insight into the evolution of player performance across different dimensions over time.

## Contributing

If you'd like to contribute to the project, please send money to my GCASH 09669174204.

## License

This project is licensed under the [MIT License](LICENSE).

## Founding Fathers

# Team Members

| Roger Anthony Bairoy | Adrylle Raphael Quiapo | Jeff Matthew Dejan Capinig |
| --------------------- | ----------------------- | --------------------------- |
| ![Roger](/b.png)      | ![Adrylle](/a.png)      | ![Jeff](/c.png)            |


## Attributes

### Rk (Rank):

- **Type:** Integer
- **Description:** Represents the rank of the player based on certain criteria. This may vary depending on the context in which the dataset was compiled.

### Player:

- **Type:** String
- **Description:** Indicates the name of the NBA player.

### Pos (Position):

- **Type:** String
- **Description:** Specifies the player's position on the basketball court (e.g., Guard, Forward, Center).

### Age:

- **Type:** Integer
- **Description:** Represents the age of the player during the specified year.

### Tm (Team):

- **Type:** String
- **Description:** Denotes the NBA team for which the player was playing during the specified year.

### G (Games Played):

- **Type:** Integer
- **Description:** Indicates the total number of games in which the player participated during the season.

### GS (Games Started):

- **Type:** Integer
- **Description:** Represents the number of games in which the player was in the starting lineup.

### MP (Minutes Played):

- **Type:** Float
- **Description:** Specifies the total number of minutes the player spent on the court during the season.

### FG (Field Goals Made):

- **Type:** Integer
- **Description:** Indicates the total number of successful field goals made by the player.

### FGA (Field Goals Attempted):

- **Type:** Integer
- **Description:** Represents the total number of field goal attempts by the player.

### FG% (Field Goal Percentage):

- **Type:** Float
- **Description:** Represents the shooting accuracy of the player, calculated as (FG / FGA) * 100.

### 3P (Three-Pointers Made):

- **Type:** Integer
- **Description:** Indicates the total number of successful three-pointers made by the player.

### 3PA (Three-Pointers Attempted):

- **Type:** Integer
- **Description:** Represents the total number of three-point attempts by the player.

### 3P% (Three-Point Percentage):

- **Type:** Float
- **Description:** Represents the shooting accuracy from three-point range, calculated as (3P / 3PA) * 100.

### 2P (Two-Pointers Made):

- **Type:** Integer
- **Description:** Indicates the total number of successful two-pointers made by the player.

### 2PA (Two-Pointers Attempted):

- **Type:** Integer
- **Description:** Represents the total number of two-point attempts by the player.

### 2P% (Two-Point Percentage):

- **Type:** Float
- **Description:** Represents the shooting accuracy from two-point range, calculated as (2P / 2PA) * 100.

### eFG% (Effective Field Goal Percentage):

- **Type:** Float
- **Description:** A composite measure of a player's shooting efficiency, considering the added value of three-pointers. Calculated as (FG + 0.5 * 3P) / FGA * 100.

### FT (Free Throws Made):

- **Type:** Integer
- **Description:** Indicates the total number of successful free throws made by the player.

### FTA (Free Throws Attempted):

- **Type:** Integer
- **Description:** Represents the total number of free throw attempts by the player.

### FT% (Free Throw Percentage):

- **Type:** Float
- **Description:** Represents the free throw shooting accuracy of the player, calculated as (FT / FTA) * 100.

### ORB (Offensive Rebounds):

- **Type:** Integer
- **Description:** Denotes the total number of offensive rebounds grabbed by the player.

### DRB (Defensive Rebounds):

- **Type:** Integer
- **Description:** Denotes the total number of defensive rebounds grabbed by the player.

### TRB (Total Rebounds):

- **Type:** Integer
- **Description:** Represents the total number of rebounds (offensive + defensive) grabbed by the player.

### AST (Assists):

- **Type:** Integer
- **Description:** Indicates the total number of assists made by the player.

### STL (Steals):

- **Type:** Integer
- **Description:** Denotes the total number of steals made by the player.

### BLK (Blocks):

- **Type:** Integer
- **Description:** Represents the total number of shots blocked by the player.

### TOV (Turnovers):

- **Type:** Integer
- **Description:** Indicates the total number of turnovers committed by the player.

### PF (Personal Fouls):

- **Type:** Integer
- **Description:** Denotes the total number of personal fouls committed by the player.

### PTS (Points):

- **Type:** Integer
- **Description:** Represents the total number of points scored by the player.

### Year:

- **Type:** (Specify the type, e.g., Integer, String)
- **Description:** (Specify the description for the "Year" attribute)




