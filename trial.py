import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import tkinter.font as tkf

def on_treeview_scroll(*args):
    tree.yview(*args)

def filter_by_selected_criteria():
    selected_year = year_combobox.get()
    selected_pos = pos_combobox.get()

    if df is not None:
        if selected_year and selected_pos:
            # Filter the DataFrame based on both selected year and position
            filtered_df = df[(df['Year'] == selected_year) & (df['Pos'] == selected_pos)]
        elif selected_year:
            # Filter the DataFrame based on selected year only
            filtered_df = df[df['Year'] == selected_year]
        elif selected_pos:
            # Filter the DataFrame based on selected position only
            filtered_df = df[df['Pos'] == selected_pos]
        else:
            # No filters selected, display the entire DataFrame
            filtered_df = df.copy()

        # Clear existing data in the treeview
        for item in tree.get_children():
            tree.delete(item)

        if not filtered_df.empty:
            # Insert filtered data into the table
            for i, row in filtered_df[columns_to_display].iterrows():
                tree.insert("", "end", values=list(row))
        else:
            # Show messagebox, reset filters, and refresh table
            messagebox.showinfo("No Results", "No data found for the selected criteria. Resetting filters.")
            reset_filters()


def reset_filters():
    # Clear the selected criteria and show the entire DataFrame
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
        # User canceled file dialog
        return

    loading_screen = tk.Toplevel(window)
    loading_screen.title("Loading CSV")

    # Set the loading screen as transient for the main window
    loading_screen.transient(window)

    # Calculate the position of the loading screen relative to the main window
    window.update_idletasks()
    x = window.winfo_x() + (window.winfo_width() - loading_screen.winfo_reqwidth()) // 2
    y = window.winfo_y() + (window.winfo_height() - loading_screen.winfo_reqheight()) // 2

    # Set the geometry of the loading screen
    loading_screen.geometry(f"+{x}+{y}")

    # Create a frame for a more organized layout
    frame = ttk.Frame(loading_screen)
    frame.pack(padx=20, pady=10)

    # Add a label to display a message
    label = ttk.Label(frame, text="Loading CSV, please wait...", font=('Helvetica', 12))
    label.grid(row=0, column=0, pady=10)

    # Create a progress bar with a different style
    style = ttk.Style()
    style.configure("TProgressbar", thickness=30)
    progress_bar = ttk.Progressbar(frame, mode='indeterminate', style="TProgressbar")
    progress_bar.grid(row=1, column=0, pady=10)

    # Start loading the CSV in a separate thread to avoid freezing the UI
    loading_screen.after(100, lambda: progress_bar.start(50))
    loading_screen.after(200, lambda: load_csv_async(file_path, loading_screen))

def clean_dataframe(df):
    # Add your cleaning logic here
    # For example, you can handle missing values or convert data types
    df.fillna(0, inplace=True)  # Replace missing values with 0

    # Replace 'Tm' values with desired values
    df['Tm'] = df['Tm'].replace({'SAC': 'Sacramento Kings', 'VAN': 'Vancouver Grizzlies', 'TOT': 'Toronto Raptors', 'SAS': 'San Antonio Spurs', 'DEN': 'Denver Nuggets', 'MIL': 'Milwaukee Bucks', 'CLE': 'Cleveland Cavaliers', 'ATL': 'Atlanta Hawks', 'POR': 'Portland Trail Blazers', 'BOS': 'Boston Celtics', 'ORL': 'Orlando Magic', 'UTA': 'Utah Jazz', 'DAL': 'Dallas Mavericks', 'SEA': 'Seattle SuperSonics', 'GSW': 'Golden State Warriors', 'CHH': 'Charlotte Hornets', 'MIA': 'Miami Heat', 'LAC': 'Los Angeles Clippers', 'PHI': 'Philadelphia 76ers', 'LAL': 'Los Angeles Lakers', 'NJN': 'New Jersey Nets', 'IND': 'Indiana Pacers', 'TOR': 'Toronto Raptors', 'CHI': 'Chicago Bulls', 'NYK': 'New York Knicks', 'PHO': 'Phoenix Suns', 'HOU': 'Houston Rockets', 'MIN': 'Minnesota Timberwolves', 'WAS': 'Washington Wizards', 'DET': 'Detroit Pistons', 'MEM': 'Memphis Grizzlies', 'NOH': 'New Orleans Hornets', 'CHA': 'Charlotte Bobcats', 'NOK': 'New Orleans/Oklahoma City Hornets', 'OKC': 'Oklahoma City Thunder', 'BRK': 'Brooklyn Nets', 'NOP': 'New Orleans Pelicans', 'CHO': 'Charlotte Hornets'})
    df['Pos'] = df['Pos'].replace({'PG': 'Point Guard', 'SG': 'Shooting Guard', 'SF': 'Small Forward', 'C': 'Center', 'PF': 'Power Forward', 'SG-SF': 'Shooting Guard - Small Forward', 'SG-PG': 'Shooting Guard - Point Guard', 'PF-C': 'Power Forward - Center', 'SF-SG': 'Small Forward - Shooting Guard', 'SF-PF': 'Small Forward - Power Forward', 'PF-SF': 'Power Forward - Small Forward', 'C-PF': 'Center - Power Forward', 'PG-SG': 'Point Guard - Shooting Guard', 'PG-SF': 'Point Guard - Small Forward', 'SG-PF': 'Shooting Guard - Power Forward', 'SF-C': 'Small Forward - Center'})

    return df

def refresh_table():
    # Clear existing data in the treeview
    for item in tree.get_children():
        tree.delete(item)

    # If no CSV file is loaded, do not insert data
    if df is None:
        return

    # Insert data into the table
    for i, row in df[columns_to_display].iterrows():
        tree.insert("", "end", values=list(row))

def show_selected_row_data(event):
    selected_item = tree.selection()
    if selected_item:
        selected_row_data = tree.item(selected_item, 'values')
        show_data_table(selected_row_data)

def show_data_table(row_data):
    selected_player = row_data[0]
    selected_rows = df[df['Player'] == selected_player]

    data_window = tk.Toplevel(window)
    data_window.title("Selected Player Data")

    # Create a treeview to display the selected rows and all columns
    selected_tree = ttk.Treeview(data_window, columns=all_columns, show="headings", style="Treeview")

    # Set column headings
    for col in all_columns:
        selected_tree.heading(col, text=col, anchor='w')

    # Insert data into the table
    for i, row in selected_rows.iterrows():
        selected_tree.insert("", "end", values=list(row))

    # Add a vertical scrollbar
    selected_vsb = ttk.Scrollbar(data_window, orient="vertical", command=selected_tree.yview)
    selected_vsb.pack(side='right', fill='y')
    selected_tree.configure(yscrollcommand=selected_vsb.set)

    # Style the treeview
    style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
    style.configure("Treeview", font=('Helvetica', 10))
    style.configure("Treeview", rowheight=25)  # Adjust the row height as needed

    # Configure the treeview to display all rows
    selected_tree.configure(height=15 + 1)  # +1 to include the header row

    # Update column widths based on content
    for col in all_columns:
        max_width = max(50, tkf.Font().measure(col), *[tkf.Font().measure(str(val)) for val in selected_rows[col]])
        selected_tree.column(col, width=max_width)

    # Pack the selected tree widget into the window
    selected_tree.pack(expand=True, fill='both')

def enable_controls():
    # Enable controls after uploading CSV
    reset_button['state'] = 'active'
    filter_button['state'] = 'active'
    year_combobox['state'] = 'readonly'
    pos_combobox['state'] = 'readonly'
    close_csv_button['state'] = 'active'
    upload_button['state'] = 'disabled'

def close_csv():
    global df, all_columns

    # Reset DataFrame and refresh table
    df = None
    all_columns = []

    # Reset filters
    reset_filters()

    # Show message
    messagebox.showinfo("CSV Closed", "CSV file closed successfully.")
    # Disable "Reset Filters" and "Apply Filter" buttons
    reset_button['state'] = 'disabled'
    filter_button['state'] = 'disabled'

    # Disable year and position comboboxes
    year_combobox['state'] = 'disabled'
    pos_combobox['state'] = 'disabled'
    upload_button['state'] = 'active'
    close_csv_button['state'] = 'disabled'

# Create the main window
window = tk.Tk()

# Set the title of the window
window.title("Tkinter Window with Pandas Table")

# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate window size (80% of screen width and height)
window_width = int(0.8 * screen_width)
window_height = int(0.8 * screen_height)

# Set window size and position to be centered
window.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")

# Use the 'clam' theme for a clean and simple design
style = ttk.Style()
style.theme_use("clam")

# Create a treeview with only the specified columns
columns_to_display = ['Player', 'Pos', 'Tm', 'Year']
tree = ttk.Treeview(window, columns=columns_to_display, show="headings", style="Treeview")

# Set column headings
for col in columns_to_display:
    tree.heading(col, text=col)

# Add a vertical scrollbar
vsb = ttk.Scrollbar(window, orient="vertical", command=on_treeview_scroll)
vsb.pack(side='right', fill='y')
tree.configure(yscrollcommand=vsb.set)

# Style the treeview
style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
style.configure("Treeview", font=('Helvetica', 10))
style.configure("Treeview", rowheight=25)  # Adjust the row height as needed

# Configure the treeview to display all rows
tree.configure(height=15 + 1)  # +1 to include the header row

# Pack the tree widget into the window
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
pos_combobox.pack(side='right')


search_label = ttk.Label(window, text="Search Player:")
search_label.pack(side='left', padx=(10, 0))

search_entry = ttk.Entry(window, state='disabled')
search_entry.pack(side='left', padx=10)
search_entry.bind('<KeyRelease>', search_players)  # Bind KeyRelease event to the search_players function


search_entry.bind("<KeyRelease>", search_players)

tree.bind("<<TreeviewSelect>>", show_selected_row_data)

window.mainloop()
