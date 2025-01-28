import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from flask_caching import Cache
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import re

# Create the Neon database connection string
NEON_CONNECTION_STRING = "postgresql://resale_transactions_owner:npg_Vsq6NDOWeG7u@ep-dry-wave-a1k1z2vx-pooler.ap-southeast-1.aws.neon.tech/resale_transactions?sslmode=require"

# Create the database connection using SQLAlchemy
engine = create_engine(NEON_CONNECTION_STRING, poolclass=NullPool)

query = "SELECT * FROM result_private;"

with engine.connect() as conn:
    result_private = pd.read_sql(
        sql=query,
        con=conn.connection
    )

app4 = dash.Dash(__name__)
server = app4.server

cache = Cache(app4.server, config={
    "CACHE_TYPE": "SimpleCache",  # Store data in memory
    "CACHE_DEFAULT_TIMEOUT": 300  # Cache timeout in seconds
})

floor_area_ranges = {
    "< 600 sqft": (0, 600),
    "600-850 sqft": (600, 850),
    "850-1100 sqft": (850, 1100),
    "1100-1500 sqft": (1100, 1500),
    "1500 sqft+": (1500, float('inf')),
    "All Resales": (0, float('inf'))
}

# Group property types
property_type_map = {
    "Apartment": "Apartment and Condominium",
    "Condominium": "Apartment and Condominium",
    "Executive Condominium": "Executive Condominium"
}

result_private["grouped_property_type"] = result_private["property_type"].map(property_type_map)

app4.layout = html.Div([
    html.H1("Private Resale Gain/Loss Dashboard", style={"textAlign": "center"}),

    # Dropdown for property type
    html.Div([
        html.Label("Select Property Type:"),
        dcc.Dropdown(
            id="property-type-dropdown",
            options=[{"label": k, "value": k} for k in result_private["grouped_property_type"].unique()],
            value=result_private["grouped_property_type"].unique()[0],
            searchable=True,
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto"}),

    html.Div([
        html.Label("Search by (Multiple Selections Possible):"),
        dcc.RadioItems(
            id="filter-mode-radio",
            options=[
                {"label": "Planning Area", "value": "Planning Area"},
                {"label": "Project Name", "value": "Project Name"}
            ],
            value="Planning Area",
            labelStyle={"display": "inline-block", "margin-right": "10px"}
        )
    ], style={"textAlign": "center", "marginTop": "20px"}),

    # Dropdown for planning area
    html.Div([
        html.Label("Select Planning Area (Searchable):"),
        dcc.Dropdown(
            id="planning-area-dropdown",
            value=None,
            multi=True,
            searchable=True,
            clearable=False
        )
    ], id="planning-area-container", style={"width": "50%", "margin": "0 auto", "marginTop": "10px"}),

    # Dropdown for project name
    html.Div([
        html.Label("Select Project Name (Searchable):"),
        dcc.Dropdown(
            id="project-dropdown",
            value=None,
            multi=True,
            searchable=True,
            clearable=False
        )
    ], id="project-container", style={"width": "50%", "margin": "0 auto", "marginTop": "10px", "display": "none"}),

    # Dropdown for floor area range
    html.Div([
        html.Label("Select Floor Area Range:"),
        dcc.Dropdown(
            id="floor-area-dropdown",
            value="All Resales",
            clearable=False
        )
    ], style={"width": "50%", "margin": "0 auto", "marginTop": "10px"}),

    # Display number of entries
    html.Div(id="num-entries", style={"textAlign": "center", "fontSize": "20px", "marginTop": "20px"}),

    # Summary statistics
    html.Div(id="summary-stats", style={"marginTop": "20px"}),

    # Scatter plot
    dcc.Graph(id="gain-loss-plot")
])

def remove_prefix(project_name):
    # Use regex to remove 'A ' or 'THE ' at the start of the string (case-insensitive)
    return re.sub(r'^(a|the)\s+', '', project_name, flags=re.IGNORECASE)

@app4.callback(
    [Output("planning-area-container", "style"),
     Output("project-container", "style")],
    [Input("filter-mode-radio", "value")]
)
def toggle_filter_mode(filter_mode):
    """Toggle visibility of Planning Area and Project Name containers."""
    if filter_mode == "Planning Area":
        # Show both Planning Area and Project Name dropdowns
        return {"width": "50%", "margin": "0 auto", "marginTop": "10px"}, {"width": "50%", "margin": "0 auto", "marginTop": "10px"}
    else:
        # Show only Project Name dropdown
        return {"display": "none"}, {"width": "50%", "margin": "0 auto", "marginTop": "10px"}

# Callback to update the dropdowns and plot
@app4.callback(
    [Output("planning-area-dropdown", "options"),
     Output("planning-area-dropdown", "value"),
     Output("project-dropdown", "options"),
     Output("project-dropdown", "value"),
     Output("floor-area-dropdown", "options"),
     Output("floor-area-dropdown", "value"),
     Output("gain-loss-plot", "figure"),
     Output("summary-stats", "children")],
    [Input("property-type-dropdown", "value"),
     Input("filter-mode-radio", "value"),
     Input("planning-area-dropdown", "value"),
     Input("project-dropdown", "value"),
     Input("floor-area-dropdown", "value")]
)

@cache.memoize()
def update_filters(property_type, filter_mode, planning_area, project_name, floor_area_range):
    # Filter by property type
    filtered_df = result_private[result_private["grouped_property_type"] == property_type]
    filtered_df['sold_at'] = pd.to_datetime(filtered_df['sold_at'])
    filtered_df['held_from'] = pd.to_datetime(filtered_df['held_from'])
    
    # Format the 'sold_at' and 'held_from' columns to only show the date
    filtered_df['Buy Date'] = filtered_df['held_from'].dt.strftime('%Y-%m-%d')
    filtered_df['Sell Date'] = filtered_df['sold_at'].dt.strftime('%Y-%m-%d')
    filtered_df['Project'] = filtered_df['project_name']
    
    # Extract year sold
    filtered_df['Year Sold'] = filtered_df['sold_at'].dt.year
    filtered_df['Address'] = filtered_df['address']
    filtered_df['Area (sqft)'] = filtered_df['area_sqft']
    
    # Format transaction price and gain/loss with commas
    filtered_df['Transacted Price (SGD)'] = filtered_df['transaction_price_dollars'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Area (sqft)'] = filtered_df['area_sqft'].apply(lambda x: f"{x:,.0f}")
    filtered_df['Gain/Loss (SGD)'] = filtered_df['Gain/Loss'].apply(lambda x: f"{x:,.0f}")
    
    filtered_df['sold_at'] = pd.to_datetime(filtered_df['sold_at'])
    filtered_df['held_from'] = pd.to_datetime(filtered_df['held_from'])
    filtered_df['Year-Month Sold'] = filtered_df['sold_at'].dt.to_period('M').astype(str)
    filtered_df['Gain/Loss Category'] = filtered_df['Gain/Loss'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Zero')
    )
    
    # Update planning area dropdown
    planning_area_options = sorted([{"label": area, "value": area} for area in filtered_df["Planning Area"].unique()],
                                key=lambda x: x["label"])


    if filter_mode == "Planning Area":
        if planning_area and isinstance(planning_area, list) and len(planning_area) > 0:
            filtered_df = filtered_df[filtered_df["Planning Area"].isin(planning_area)]
    else:
        planning_area = None
    
    # Update project name dropdown
    project_options = sorted([{"label": name, "value": name} for name in filtered_df["project_name"].unique()],
                          key=lambda x: remove_prefix(x["label"]))


    if project_name and isinstance(project_name, list) and len(project_name) > 0:
        filtered_df = filtered_df[filtered_df["project_name"].isin(project_name)]

    if filter_mode == "Project Name" and project_name:
        filtered_df = filtered_df[filtered_df["project_name"].isin(project_name)]

    # Update floor area dropdown
    min_area = filtered_df["area_sqft"].min()
    max_area = filtered_df["area_sqft"].max()
    available_floor_ranges = {key: (min_val, max_val) for key, (min_val, max_val) in floor_area_ranges.items()
                              if min_area <= max_val and max_area >= min_val}
    floor_area_options = [{"label": key, "value": key} for key in available_floor_ranges.keys()]
    default_floor_area = floor_area_range if floor_area_range in available_floor_ranges else "All Resales"

    # Filter by floor area range
    floor_min, floor_max = available_floor_ranges.get(default_floor_area, (0, float('inf')))
    filtered_df = filtered_df[(filtered_df["area_sqft"] >= floor_min) & (filtered_df["area_sqft"] <= floor_max)]

    
    if not filtered_df.empty:
        max_months = filtered_df["months_held"].max()
        median_months = filtered_df["months_held"].median()
        min_months = filtered_df["months_held"].min()
    else:
        max_months = median_months = min_months = 0

    # Convert months to "years and months" format
    def format_years_months(total_months):
        years = total_months // 12
        months = total_months % 12
        return f"{years} years {months} months" if years > 0 else f"{months} months"

    max_period = format_years_months(max_months)
    median_period = format_years_months(int(median_months))  # Ensure median is an integer
    min_period = format_years_months(min_months)

    num_gains = (filtered_df['Gain/Loss'] > 0).sum()
    num_losses = (filtered_df['Gain/Loss'] < 0).sum()
    num_zero = (filtered_df['Gain/Loss'] == 0).sum()    
    # Calculate Gain/Loss statistics
    largest_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].max() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    largest_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].min() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0
    median_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].median() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    median_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].median() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0
    smallest_gain = filtered_df[filtered_df['Gain/Loss'] > 0]['Gain/Loss'].min() if not filtered_df[filtered_df['Gain/Loss'] > 0].empty else 0
    smallest_loss = filtered_df[filtered_df['Gain/Loss'] < 0]['Gain/Loss'].max() if not filtered_df[filtered_df['Gain/Loss'] < 0].empty else 0

    total_gains = (filtered_df['Gain/Loss'] > 0).sum()
    total_losses = (filtered_df['Gain/Loss'] < 0).sum()
    total_resales = len(filtered_df)
    percentage_gains = total_gains / total_resales * 100
    percentage_loss = total_losses / total_resales * 100
    percentage_zero = 100 - (percentage_gains + percentage_loss)
    num_entries = len(filtered_df)
    # Summary statistics
    summary_stats = html.Div([
        html.Hr(),
        html.Div(f"Number of Resales: {num_entries}", style={"marginBottom": "5px"}),
        html.Div(f"Gains: {num_gains} ({percentage_gains:.2f}%)", style={"marginBottom": "5px"}),
        html.Div(f"Losses: {num_losses} ({percentage_loss:.2f}%)", style={"marginBottom": "5px"}),
        html.Div(f"Sold at same price: {num_zero} ({percentage_zero:.2f}%)", style={"marginBottom": "5px"}),
        html.Hr(),
        html.Div(f"Max Holding Period: {max_period}", style={"marginBottom": "5px"}),
        html.Div(f"Median Holding Period: {median_period}", style={"marginBottom": "5px"}),
        html.Div(f"Min Holding Period: {min_period}", style={"marginBottom": "5px"}),
        html.Hr(),

        # Grouping Largest Gain/Loss, Median Gain/Loss, Smallest Gain/Loss
        html.Div(f"Median Gain/Loss: {filtered_df['Gain/Loss'].median():,.0f} SGD", style={"marginBottom": "5px"}),
        html.Div([
            html.Div(f"Largest Gain: {largest_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Largest Loss: {largest_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Div(f"Median Gain: {median_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Median Loss: {median_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Div(f"Smallest Gain: {smallest_gain:,.0f} SGD", style={"width": "45%", "display": "inline-block"}),
            html.Div(f"Smallest Loss: {smallest_loss:,.0f} SGD", style={"width": "45%", "display": "inline-block"})
        ], style={"marginBottom": "5px"}),

        html.Hr()  # Add a line after Smallest Gain/Smallest Loss
    ], style={"textAlign": "center", "fontSize": "18px"})    
    
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Zero': 'yellow'
    }    

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="Year-Month Sold",
        y="Gain/Loss",
        hover_data={
            "Gain/Loss (SGD)": True,
            "Buy Date": True,
            "Sell Date": True,
            "Area (sqft)": True,
            "Project": True,
            "Address": True,
            "Original Price (SGD)": True,
            "Transacted Price (SGD)": True,
            "Gain/Loss": False,
            "Gain/Loss Category": False
        },
        color="Gain/Loss Category",
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        transition_duration=500,
        height=800,
        yaxis=dict(
            tickformat=',.0f'
        )
    )


    # Create summary stats 


    return (planning_area_options, planning_area,
            project_options, project_name,
            floor_area_options, default_floor_area,
            fig, summary_stats)


if __name__ == "__main__":
    app4.run_server(port=8056, debug=True)