import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

def on_treeview_scroll(*args):
    tree.yview(*args)

def filter_by_selected_criteria():
    selected_year = year_combobox.get()
    selected_pos = pos_combobox.get()

    if df is not None and selected_year and selected_pos:
        # Filter the DataFrame based on selected year and position
        filtered_df = df[(df['Year'] == selected_year) & (df['Pos'] == selected_pos)]

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

def upload_csv():
    global df, all_columns
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)

        # Clean the DataFrame (replace this with your cleaning logic)
        df = clean_dataframe(df)

        # Get all columns from the CSV file
        all_columns = list(df.columns)

        refresh_table()
        enable_controls()  # Enable controls after uploading CSV

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

    # Insert data into the table
    if df is not None:
        for i, row in df[columns_to_display].iterrows():
            tree.insert("", "end", values=list(row))

def show_selected_row_data(event):
    selected_item = tree.selection()
    if selected_item:
        selected_row_data = tree.item(selected_item, 'values')
        show_data_window(selected_row_data)

def show_data_window(row_data):
    selected_row = df[(df['Player'] == row_data[0]) & (df['Pos'] == row_data[1]) & (df['Tm'] == row_data[2]) & (df['Year'] == row_data[3])]
    data_window = tk.Toplevel(window)
    data_window.title("Selected Row Data")

    data_text = tk.Text(data_window, wrap="none", height=len(all_columns), width=50)

    # Iterate over all columns and concatenate values
    all_data = ""
    for col in all_columns:
        all_data += f"{col}: {selected_row[col].values[0]}\n"

    data_text.insert("1.0", all_data)
    data_text.pack()

def enable_controls():
    # Enable controls after uploading CSV
    reset_button['state'] = 'active'
    filter_button['state'] = 'active'
    year_combobox['state'] = 'readonly'
    pos_combobox['state'] = 'readonly'

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

# Create a Pandas DataFrame widget
df = None
all_columns = []

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

# Create Comboboxes for selecting the year and position
year_combobox = ttk.Combobox(window, values=[
    '1998-1999', '1999-2000', '2000-2001', '2001-2002', '2002-2003', '2003-2004',
    '2004-2005', '2005-2006', '2006-2007', '2007-2008', '2008-2009', '2009-2010',
    '2010-2011', '2011-2012', '2012-2013', '2013-2014', '2014-2015', '2015-2016',
    '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022'
])

pos_combobox = ttk.Combobox(window, values=[
    'Point Guard', 'Shooting Guard', 'Small Forward', 'Center', 'Power Forward',
    'Shooting Guard - Small Forward', 'Shooting Guard - Point Guard',
    'Power Forward - Center', 'Small Forward - Shooting Guard',
    'Small Forward - Power Forward', 'Power Forward - Small Forward',
    'Center - Power Forward', 'Point Guard - Shooting Guard',
    'Point Guard - Small Forward', 'Shooting Guard - Power Forward',
    'Small Forward - Center']
)

# Create buttons and combobox on the right side
upload_button = ttk.Button(window, text="Upload CSV", command=upload_csv)
upload_button.pack(side='right')

reset_button = ttk.Button(window, text="Reset Filters", command=reset_filters, state='disabled')
reset_button.pack(side='right')

filter_button = ttk.Button(window, text="Apply Filter", command=filter_by_selected_criteria, state='disabled')
filter_button.pack(side='right')

year_combobox.pack(side='right')
pos_combobox.pack(side='right')

# Bind the TreeviewSelect event to show the data details
tree.bind("<<TreeviewSelect>>", show_selected_row_data)

# Start the Tkinter event loop
window.mainloop()
