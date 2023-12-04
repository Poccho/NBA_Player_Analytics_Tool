import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import mplcursors


def on_close():
    window.destroy()
def train_prediction_model():
    global df

    # Check if the dataframe is not empty
    if df is not None:
        # Assume 'Year' is the predictor variable
        predictor_column = 'Year'

        if predictor_column in df.columns:
            # Select features (excluding the predictor column and non-numerical columns)
            non_numerical_columns = ['Age', 'Tm', 'Pos', 'Player']
            features = df.drop([predictor_column] + non_numerical_columns, axis=1)

            # Select the target variable (all columns except the non-numerical ones)
            target_variable = df.drop(non_numerical_columns, axis=1).columns

            if not target_variable.empty:

                # Assuming 'target_variable' is the name of your target column
                df[target_variable] = pd.to_numeric(df[target_variable].iloc[:, 0], errors='coerce')

                print(len(df[target_variable].iloc[:, 0]))
                print(len(df[target_variable]))

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    features, df[target_variable], test_size=0.2, random_state=42)

                # Train a linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                messagebox.showinfo("Model Trained", "Prediction model trained successfully.")
            else:
                messagebox.showerror("Error", "No target variable found in the dataset.")
        else:
            messagebox.showerror("Error", "Predictor column not found.")
    else:
        messagebox.showerror("Error", "No data available. Please upload a CSV file.")


def on_treeview_scroll(*args):
    tree.yview(*args)

def filter_by_selected_criteria():
    selected_year = year_combobox.get()
    selected_pos = pos_combobox.get()

    if df is not None:
        if selected_year and selected_pos:
            filtered_df = df[(df['Year'] == selected_year) & (df['Pos'] == selected_pos)]
        elif selected_year:
            filtered_df = df[df['Year'] == selected_year]
        elif selected_pos:
            filtered_df = df[df['Pos'] == selected_pos]
        else:
            filtered_df = df.copy()

        for item in tree.get_children():
            tree.delete(item)

        if not filtered_df.empty:
            for i, row in filtered_df[columns_to_display].iterrows():
                tree.insert("", "end", values=list(row))
        else:
            messagebox.showinfo("No Results", "No data found for the selected criteria. Resetting filters.")
            reset_filters()

def reset_filters():
    year_combobox.set('')
    pos_combobox.set('')
    refresh_table()

def load_csv_async(file_path, loading_screen):
    global df, all_columns

    if file_path:
        df = pd.read_csv(file_path)
        df = clean_dataframe(df)
        all_columns = list(df.columns)
        refresh_table()
        enable_controls()
        loading_screen.destroy()

def upload_csv():
    global df, all_columns

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not file_path:
        return

    loading_screen = tk.Toplevel(window)
    loading_screen.title("Loading CSV")

    loading_screen.transient(window)
    window.update_idletasks()
    x = window.winfo_x() + (window.winfo_width() - loading_screen.winfo_reqwidth()) // 2
    y = window.winfo_y() + (window.winfo_height() - loading_screen.winfo_reqheight()) // 2

    loading_screen.geometry(f"+{x}+{y}")

    frame = ttk.Frame(loading_screen)
    frame.pack(padx=20, pady=10)

    label = ttk.Label(frame, text="Loading CSV, please wait...", font=('Helvetica', 12))
    label.grid(row=0, column=0, pady=10)

    style = ttk.Style()
    style.configure("TProgressbar", thickness=30)
    progress_bar = ttk.Progressbar(frame, mode='indeterminate', style="TProgressbar")
    progress_bar.grid(row=1, column=0, pady=10)

    loading_screen.after(100, lambda: progress_bar.start(50))
    loading_screen.after(200, lambda: load_csv_async(file_path, loading_screen))

def clean_dataframe(df):
    df.fillna(0, inplace=True)
    df['Tm'] = df['Tm'].replace({'SAC': 'Sacramento Kings', 'VAN': 'Vancouver Grizzlies', 'TOT': 'Toronto Raptors', 'SAS': 'San Antonio Spurs', 'DEN': 'Denver Nuggets', 'MIL': 'Milwaukee Bucks', 'CLE': 'Cleveland Cavaliers', 'ATL': 'Atlanta Hawks', 'POR': 'Portland Trail Blazers', 'BOS': 'Boston Celtics', 'ORL': 'Orlando Magic', 'UTA': 'Utah Jazz', 'DAL': 'Dallas Mavericks', 'SEA': 'Seattle SuperSonics', 'GSW': 'Golden State Warriors', 'CHH': 'Charlotte Hornets', 'MIA': 'Miami Heat', 'LAC': 'Los Angeles Clippers', 'PHI': 'Philadelphia 76ers', 'LAL': 'Los Angeles Lakers', 'NJN': 'New Jersey Nets', 'IND': 'Indiana Pacers', 'TOR': 'Toronto Raptors', 'CHI': 'Chicago Bulls', 'NYK': 'New York Knicks', 'PHO': 'Phoenix Suns', 'HOU': 'Houston Rockets', 'MIN': 'Minnesota Timberwolves', 'WAS': 'Washington Wizards', 'DET': 'Detroit Pistons', 'MEM': 'Memphis Grizzlies', 'NOH': 'New Orleans Hornets', 'CHA': 'Charlotte Bobcats', 'NOK': 'New Orleans/Oklahoma City Hornets', 'OKC': 'Oklahoma City Thunder', 'BRK': 'Brooklyn Nets', 'NOP': 'New Orleans Pelicans', 'CHO': 'Charlotte Hornets'})
    df['Pos'] = df['Pos'].replace({'PG': 'Point Guard', 'SG': 'Shooting Guard', 'SF': 'Small Forward', 'C': 'Center', 'PF': 'Power Forward', 'SG-SF': 'Shooting Guard - Small Forward', 'SG-PG': 'Shooting Guard - Point Guard', 'PF-C': 'Power Forward - Center', 'SF-SG': 'Small Forward - Shooting Guard', 'SF-PF': 'Small Forward - Power Forward', 'PF-SF': 'Power Forward - Small Forward', 'C-PF': 'Center - Power Forward', 'PG-SG': 'Point Guard - Shooting Guard', 'PG-SF': 'Point Guard - Small Forward', 'SG-PF': 'Shooting Guard - Power Forward', 'SF-C': 'Small Forward - Center'})
    return df

def refresh_table():
    for item in tree.get_children():
        tree.delete(item)

    if df is not None:
        for i, row in df[columns_to_display].iterrows():
            tree.insert("", "end", values=list(row))

def show_selected_row_data(event):
    selected_item = tree.selection()
    if selected_item:
        selected_row_data = tree.item(selected_item, 'values')
        show_data_table(selected_row_data)


def update_radar_chart(selected_player, selected_year, ax, canvas, data_window, filtered_rows=None):
    global selected_rows

    # Clear the existing radar chart
    ax.clear()

    if filtered_rows is None:
        # Use the original data if no filtered data is provided
        filtered_rows = selected_rows

    # Assuming these are your columns of interest
    radar_columns = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    # Get unique teams for the selected player and year
    unique_teams = filtered_rows[(filtered_rows['Year'] == selected_year) & (filtered_rows['Player'] == selected_player)]['Tm'].unique()

    # Define a colormap for teams
    team_colors = plt.cm.get_cmap('tab10', len(unique_teams))

    # Plot the radar chart for each team
    for i, team in enumerate(unique_teams):
        team_filtered_rows = filtered_rows[(filtered_rows['Year'] == selected_year) & (filtered_rows['Player'] == selected_player) & (filtered_rows['Tm'] == team)]
        values = team_filtered_rows[radar_columns].values.flatten().tolist()
        values += values[:1]  # Close the plot

        # Use a different color for each team
        color = team_colors(i)

        # Plot the radar chart line with a label for the team
        ax.plot(np.linspace(0, 2 * np.pi, len(values)), values, label=f"{team} ({selected_year})", color=color)

        # Plot the values inside the radar chart
        for j, value in enumerate(values):
            angle = (j / len(values)) * 2 * np.pi
            ax.text(angle, value, f"{value:.2f}", color=color, ha='center', va='bottom', fontsize=10)

    # Connect the data points with lines
    ax.plot(np.linspace(0, 2 * np.pi, len(values)), values, color='black', linewidth=1, linestyle='solid')

    # Set the labels for each axis
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(values))[:-1])
    ax.set_xticklabels(radar_columns)

    # Add labels and legend
    ax.set_title(f"{selected_player}'s Data for {selected_year}")
    ax.legend(loc='upper right')

    # Update the canvas
    canvas.draw()


details_window = None


def show_data_table(row_data):
    global selected_rows, data_window, canvas

    def show_more_details():
        global details_window  # Declare details_window as a global variable

        # Check if details_window exists and destroy it if it does
        if details_window:
            details_window.destroy()

        selected_year = areachart_year.get()

        details_window = tk.Toplevel(window)
        details_window.title(f"Player Details - {selected_year}")

        # Filter rows based on the selected year
        details_rows = selected_rows[
            (selected_rows['Player'] == selected_player) & (selected_rows['Year'] == selected_year)]

        # Check if there is data available
        if details_rows.empty:
            messagebox.showinfo("No Data", "No data available for the selected player and year.")
            details_window.destroy()
            return

        # Set columns to display in the radar chart
        radar_columns = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
                         'DRB', 'TRB', 'AST', 'STL', 'BLK']

        # Create a Figure and Axis for the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Normalize data to be between 0 and 1 for radar chart
        values = details_rows[radar_columns].values.flatten().tolist()
        # Number of data points
        num_columns = len(radar_columns)

        # Compute angles for each axis
        angles = np.linspace(0, 2 * np.pi, num_columns, endpoint=False).tolist()

        # The plot is circular, so we need to "close the loop" and connect the first and last points
        # values = normalized_data.flatten().tolist()
        values += values[:1]
        angles += angles[:1]

        # Ensure both angles and values have the same first dimension
        if len(angles) != len(values):
            values = values[:len(angles)]

        # Define a colormap for teams
        team_colors = plt.cm.get_cmap('tab10', len(details_rows['Tm'].unique()))

        # Plot the data
        for i, team in enumerate(details_rows['Tm'].unique()):
            team_filtered_rows = details_rows[details_rows['Tm'] == team]
            team_values = team_filtered_rows[radar_columns].values.flatten().tolist()
            team_values += team_values[:1]  # Close the plot

            # Use a different color for each team
            color = team_colors(i)

            # Plot the radar chart line with a label for the team
            ax.plot(angles, team_values, label=f"{team} ({selected_year})", color=color, linewidth=2)

            # Plot the values inside the radar chart
            for j, value in enumerate(team_values):
                angle = (j / len(values)) * 2 * np.pi
                ax.text(angle, value, f"{value:.2f}", color=color, ha='center', va='bottom', fontsize=10)

        # Connect the data points with lines
        ax.plot(angles, values, color='black', linewidth=1, linestyle='solid')

        # Set the labels for each axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_columns)

        # Add labels and legend
        ax.set_title(f"{selected_player}'s Data for {selected_year}")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        # Create a Tkinter canvas to embed the Matplotlib chart
        canvas = FigureCanvasTkAgg(fig, master=details_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill='both')

        # Draw the chart on the canvas
        canvas.draw()


    selected_player = row_data[0]
    selected_rows = df[df['Player'] == selected_player]

    # Filter out empty values in the 'Year' column
    unique_years = selected_rows['Year'].dropna().unique()
    unique_years_str = list(map(str, unique_years))

    data_window = tk.Toplevel(window)
    data_window.title("Selected Player Data")

    # Create frames for left and right sides
    left_frame = ttk.Frame(data_window)
    left_frame.pack(side='left', padx=10)

    right_frame = ttk.Frame(data_window)
    right_frame.pack(side='left', padx=10)

    # Set a custom style for the labels
    style = ttk.Style()
    style.configure('Label.TLabel', font=('Helvetica', 14))

    title_label = ttk.Label(left_frame, text=f"Player: {selected_player}", style='Label.TLabel')
    title_label.pack(pady=10)

    areachart_year = ttk.Combobox(left_frame, state='readonly', values=unique_years_str)
    areachart_year.pack(pady=10)
    areachart_year.set(str(unique_years.max()))  # Set default value to the latest year

    rank_label = ttk.Label(left_frame, text="Rank:", style='Label.TLabel')
    rank_label.pack(padx=(0, 10), anchor='w')  # 'w' stands for west, aligning to the left

    age_label = ttk.Label(left_frame, text="Age:", style='Label.TLabel')
    age_label.pack(padx=(0, 10), anchor='w')

    pos_label = ttk.Label(left_frame, text="Position:", style='Label.TLabel')
    pos_label.pack(padx=(0, 10), anchor='w')

    team_label = ttk.Label(left_frame, text="Team:", style='Label.TLabel')
    team_label.pack(padx=(0, 10), anchor='w')

    g_label = ttk.Label(left_frame, text="G:", style='Label.TLabel')
    g_label.pack(padx=(0, 10), anchor='w')

    gs_label = ttk.Label(left_frame, text="GS:", style='Label.TLabel')
    gs_label.pack(padx=(0, 10), anchor='w')

    mp_label = ttk.Label(left_frame, text="MP:", style='Label.TLabel')
    mp_label.pack(padx=(0, 10), anchor='w')

    tov_label = ttk.Label(left_frame, text="TOV:", style='Label.TLabel')
    tov_label.pack(padx=(0, 10), anchor='w')

    pf_label = ttk.Label(left_frame, text="PF:", style='Label.TLabel')
    pf_label.pack(padx=(0, 10), anchor='w')

    pts_label = ttk.Label(left_frame, text="PTS:", style='Label.TLabel')
    pts_label.pack(padx=(0, 10), anchor='w')

    more_details_button = ttk.Button(left_frame, text="More Details", command=show_more_details)
    more_details_button.pack(pady=10)

    # Check if the latest year is 2021-2022
    if '2021-2022' in unique_years_str:
        predict_button = ttk.Button(left_frame, text="Predict Stats",
                                    command=lambda: predict_stats(selected_player))
        predict_button.pack(pady=10)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(expand=True, fill='both')

    # Draw the chart on the canvas
    filter_by_selected_year(selected_player, areachart_year.get(), canvas, ax, rank_label, pos_label, age_label,
                            team_label, g_label, gs_label, mp_label, tov_label, pf_label, pts_label)

    # Bind the filter_by_selected_year function to the ComboboxSelected event
    areachart_year.bind("<<ComboboxSelected>>",
                        lambda event: filter_by_selected_year(selected_player, areachart_year.get(), canvas, ax,
                                                              rank_label, pos_label, age_label, team_label, g_label,
                                                              gs_label, mp_label, tov_label, pf_label, pts_label))


def calculate_total_stats():
    if df is not None:
        # Retrieve the selected category from the Combobox
        selected_category = category_var.get()

        # Check if the selected column is numeric before calculating the total
        if pd.api.types.is_numeric_dtype(df[selected_category]):
            total_stats = df.groupby('Player')[selected_category].sum()

            # Filter out players with a total value less than 1
            total_stats = total_stats[total_stats >= 1]

            if not total_stats.empty:
                # Create a new window to display scatter plots
                total_stats_window = tk.Toplevel(window)
                total_stats_window.title(f'Total Stats Scatter Plots - {selected_category}')

                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = sns.scatterplot(x=total_stats.index, y=total_stats, marker='o', color='blue')

                # Use mplcursors to display annotations on hover
                mplcursors.cursor(scatter, hover=True).connect(
                    "add",
                    lambda sel: sel.annotation.set_text(f"{df['Player'][sel.target.index]}\n{selected_category}: {sel.target[1]:.2f}"))

                plt.title(f'Total {selected_category} for each player')
                plt.ylabel(selected_category)
                ax.set_xticklabels([])  # Remove x-axis labels
                plt.xticks(rotation=90)
                plt.xlabel('')  # Set an empty string for x-axis label
                plt.tight_layout()

                # Embed the matplotlib figure in the Tkinter window
                canvas = FigureCanvasTkAgg(fig, master=total_stats_window)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(expand=True, fill='both')

def predict_stats(selected_player):
        # Filter rows for the selected player
        player_rows = selected_rows[selected_rows['Player'] == selected_player]

        # Check if there are enough samples for splitting
        if len(player_rows) < 2:
            messagebox.showinfo("Insufficient Data", "Insufficient data to predict stats.")
            return

        # Features and target variable
        columns_to_predict = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
                              'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        features = player_rows[columns_to_predict]
        target = player_rows[columns_to_predict]  # Change the target variable to include all columns

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Create a linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Use the model to predict future values
        future_features = np.array([[row[col] for col in columns_to_predict] for _, row in player_rows.iterrows()])
        future_predictions = model.predict(future_features)

        # Create a new window for predictions
        prediction_window = tk.Toplevel(window)
        prediction_window.title(f"Predicted Stats - {selected_player} (2022-2023)")

        # Get the screen width and height
        screen_width = prediction_window.winfo_screenwidth()
        screen_height = prediction_window.winfo_screenheight()

        # Calculate the position for centering the window
        x_position = (screen_width - prediction_window.winfo_reqwidth()) / 2
        y_position = (screen_height - prediction_window.winfo_reqheight()) / 2

        # Set the window position
        prediction_window.geometry("+%d+%d" % (x_position, y_position))

        # Create a text widget
        text_widget = tk.Text(prediction_window, wrap=tk.WORD, width=30, height=10)
        text_widget.grid(row=1, column=0, padx=5, pady=5)

        # Display the predictions in the text widget
        text_widget.insert(tk.END, "Column\tPrediction\n")
        for col, prediction in zip(columns_to_predict, future_predictions[0]):
            text_widget.insert(tk.END, f"{col}\t{prediction:.2f}\n")

def filter_by_selected_year(selected_player, selected_year, canvas, ax, rank_label, pos_label, age_label, team_label,
                             g_label, gs_label, mp_label, tov_label, pf_label, pts_label):
    global selected_rows, data_window

    # Filter the data for the selected year
    filtered_rows = selected_rows[
        (selected_rows['Year'] == selected_year) & (selected_rows['Player'] == selected_player)]

    # Clear the existing radar chart
    ax.clear()

    if not filtered_rows.empty:
        # Assuming these are your columns of interest
        radar_columns = ['3P%', '2P%', 'AST', 'FT', 'PTS']

        # Plot the radar chart for the filtered data
        for i, (_, row) in enumerate(filtered_rows.iterrows()):
            values = row[radar_columns].values.flatten().tolist()
            values += values[:1]  # Close the plot
            color = plt.cm.get_cmap('tab10')(i)  # Get a distinct color for each team
            ax.plot(np.linspace(0, 2 * np.pi, len(values)), values, label=row['Tm'], color=color)

        # Set the labels for each axis
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(values))[:-1])
        ax.set_xticklabels(radar_columns)

        # Add labels and legend
        ax.set_title(f"{selected_player}'s Data for {selected_year}")
        ax.legend(loc='upper right')

        # Update the canvas
        canvas.draw()

        # Display Rank, Position, Age, and Team information
        rank_label.config(text=f"Rank: {filtered_rows['Rk'].values[0]}")
        pos_label.config(text=f"Position: {filtered_rows['Pos'].values[0]}")
        age_label.config(text=f"Age: {filtered_rows['Age'].values[0]}")

        # Display teams in a column
        teams = '\n'.join(filtered_rows['Tm'].astype(str))
        team_label.config(text=f"Team:\n{teams}")

        # Display additional columns
        g_label.config(text=f"Games played: {'// '.join(filtered_rows['G'].astype(str))}")
        gs_label.config(text=f"Games started: {'// '.join(filtered_rows['GS'].astype(str))}")
        mp_label.config(text=f"Minutes per game: {'// '.join(filtered_rows['MP'].astype(str))}")
        tov_label.config(text=f"Turnovers: {'// '.join(filtered_rows['TOV'].astype(str))}")
        pf_label.config(text=f"Personal fouls: {'// '.join(filtered_rows['PF'].astype(str))}")
        pts_label.config(text=f"Points: {'// '.join(filtered_rows['PTS'].astype(str))}")

    else:
        ttk.Label(data_window, text=f"No data available for {selected_year}.", font=('Helvetica', 12)).pack(pady=10)

def enable_controls():
    reset_button['state'] = 'active'
    filter_button['state'] = 'active'
    year_combobox['state'] = 'readonly'
    pos_combobox['state'] = 'readonly'
    close_csv_button['state'] = 'active'
    upload_button['state'] = 'disabled'
    search_entry['state'] = 'active'
    category_combobox['state'] = 'readonly'
    avg_stats_button['state'] = 'active'


def close_csv():
    global df, all_columns
    df = None
    all_columns = []
    reset_filters()
    messagebox.showinfo("CSV Closed", "CSV file closed successfully.")
    reset_button['state'] = 'disabled'
    filter_button['state'] = 'disabled'
    year_combobox['state'] = 'disabled'
    pos_combobox['state'] = 'disabled'
    upload_button['state'] = 'active'
    close_csv_button['state'] = 'disabled'
    search_entry['state'] = 'disabled'
    category_combobox['state'] = 'disabled'
    avg_stats_button['state'] = 'disabled'



error_message_shown = False
search_delay = 1000  # 1000 milliseconds (1 second)

after_id = None

def search_players(event):
    global after_id
    if after_id:
        window.after_cancel(after_id)
    after_id = window.after(search_delay, perform_search)


def perform_search():
    global error_message_shown

    search_term = search_entry.get()

    if not search_term:
        # If the search term is empty, refresh the table with the original data
        refresh_table()
        return

    search_result = df[df['Player'].str.contains(search_term, case=False)]

    if not search_result.empty:
        for item in tree.get_children():
            tree.delete(item)

        for i, row in search_result[columns_to_display].iterrows():
            tree.insert("", "end", values=list(row))

        # Reset the error message status
        error_message_shown = False
    else:
        # Show error message only if it hasn't been shown before
        if not error_message_shown:
            messagebox.showinfo("No Results", f"No data found for '{search_term}'. Resetting search.")
            error_message_shown = True
            # Reset the search entry only when an error message is shown
            search_entry.delete(0, tk.END)
            reset_filters()


# Create the main window
window = tk.Tk()

window.title("NBA PLAYER STATS")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window_width = int(0.8 * screen_width)
window_height = int(0.8 * screen_height)

window.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")

style = ttk.Style()
style.theme_use("clam")

columns_to_display = ['Player', 'Pos', 'Tm', 'Year']
tree = ttk.Treeview(window, columns=columns_to_display, show="headings", style="Treeview")

for col in columns_to_display:
    tree.heading(col, text=col)

vsb = ttk.Scrollbar(window, orient="vertical", command=on_treeview_scroll)
vsb.pack(side='right', fill='y')
tree.configure(yscrollcommand=vsb.set)

style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
style.configure("Treeview", font=('Helvetica', 10))
style.configure("Treeview", rowheight=25)

tree.configure(height=15 + 1)

tree.pack(expand=True, fill='both')

year_combobox = ttk.Combobox(window, state='disabled', values=[
    '1998-1999', '1999-2000', '2000-2001', '2001-2002', '2002-2003', '2003-2004',
    '2004-2005', '2005-2006', '2006-2007', '2007-2008', '2008-2009', '2009-2010',
    '2010-2011', '2011-2012', '2012-2013', '2013-2014', '2014-2015', '2015-2016',
    '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022'
])

pos_combobox = ttk.Combobox(window, state='disabled', values=[
    'Point Guard', 'Shooting Guard', 'Small Forward', 'Center', 'Power Forward',
    'Shooting Guard - Small Forward', 'Shooting Guard - Point Guard',
    'Power Forward - Center', 'Small Forward - Shooting Guard',
    'Small Forward - Power Forward', 'Power Forward - Small Forward',
    'Center - Power Forward', 'Point Guard - Shooting Guard',
    'Point Guard - Small Forward', 'Shooting Guard - Power Forward',
    'Small Forward - Center']
)

close_csv_button = ttk.Button(window, text="Close CSV", command=close_csv, state='disabled')
close_csv_button.pack(side='right')

upload_button = ttk.Button(window, text="Upload CSV", command=upload_csv)
upload_button.pack(side='right')

reset_button = ttk.Button(window, text="Reset Filters", command=reset_filters, state='disabled')
reset_button.pack(side='right')

filter_button = ttk.Button(window, text="Apply Filter", command=filter_by_selected_criteria, state='disabled')
filter_button.pack(side='right')

year_combobox.pack(side='right')
year_label = ttk.Label(window, text="Years:")
year_label.pack(side='right', padx=(10, 0))


pos_combobox.pack(side='right')
pos_label = ttk.Label(window, text="Position:")
pos_label.pack(side='right', padx=(10, 0))

avg_stats_button = ttk.Button(window, text="Plot Average Stats", command=calculate_total_stats, state='disabled')
avg_stats_button.pack(side='left', padx=10)

categories = ['G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
              'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

category_var = tk.StringVar()
category_combobox = ttk.Combobox(window, textvariable=category_var, values=categories, state="disabled")
category_combobox.set(categories[0])  # Set the default value
category_combobox.pack(side='left', padx=10)

search_label = ttk.Label(window, text="Search Player:")
search_label.pack(side='right', padx=(10, 0))

search_entry = ttk.Entry(window, state='disabled')
search_entry.pack(side='right', padx=10)
search_entry.bind('<KeyRelease>', search_players)  # Bind KeyRelease event to the search_players function


search_entry.bind("<KeyRelease>", search_players)

tree.bind("<<TreeviewSelect>>", show_selected_row_data)

window.protocol("WM_DELETE_WINDOW", on_close)

window.mainloop()
