# --- World Happiness Agent ---

from pathlib import Path
import os
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Folder containing the yearly CSV files
DATA_DIR = Path(__file__).parent / "happiness_project"

# Shared DataFrame used by the tools
df = None

# --- Tool 1: Load Happiness Data ---

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    
    """Load and combine all yearly World Happiness CSV files.

    Reads the yearly CSV files from the happiness_project folder,
    standardizes the required column names, adds a year column,
    and combines all years into one DataFrame.

    Args:
        None.

    Returns:
        A dictionary containing the DataFrame shape and column names.
        Returns an error dictionary if no files are found.
    """
    global df

    frames = []

    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    if not frames:
        return {"error": "No World Happiness CSV files were found."}

    df = pd.concat(frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

# --- Tool 2: Summarize a Column ---

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a dataset column.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary containing descriptive statistics, or an error
        message if the data or column is unavailable.
    """
    if df is None:
        return {"error": "Please load the happiness data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    return df[column].describe().to_dict()

# --- Tool 3: Compute Correlation ---

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Calculate the Pearson correlation between two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing both column names, Pearson correlation
        coefficient, and p-value, rounded to four decimal places.
    """
    if df is None:
        return {"error": "Please load the happiness data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns were not found."}

    try:
        data = df[[col1, col2]].dropna()

        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in [col1, col2]):
            return {"error": "Both columns must contain numeric data."}

        pearson_r, p_value = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(pearson_r, 4),
            "p_value": round(p_value, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# --- Tool 4: Get Top N Countries ---

# --- Tool 4: Get Top N Countries ---

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a column for a specific year.

    Args:
        column: Name of the column used to rank countries.
        year: Year to filter the dataset.
        n: Number of top countries to return. Defaults to 5.

    Returns:
        A list of dictionaries containing each country's name and
        its value for the requested column, or an error dictionary
        if the input is invalid or the data is unavailable.
    """
    if df is None:
        return {"error": "Please load the happiness data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    if not isinstance(year, int) or year not in df["year"].values:
        return {"error": f"Invalid year: {year}."}

    if not isinstance(n, int) or n < 1:
        return {"error": "n must be a positive integer."}

    year_data = df[df["year"] == year]

    top_countries = (
        year_data
        .dropna(subset=[column])
        .sort_values(by=column, ascending=False)
        .head(n)
    )

    return [
        {
            "country": row["country"],
            column: row[column]
        }
        for _, row in top_countries.iterrows()
    ]

# --- Task 2: Build the CodeAgent ---

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns,
computing correlations, and ranking countries.
Write Python code directly only when the tools are not sufficient,
for example, when creating custom plots or computing something
the tools do not cover.
The load_happiness_data tool returns a dictionary, not a DataFrame.
Access its results using the keys 'shape' and 'columns'.
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats",
        "glob"
    ],
    max_steps=5
)

# --- Task 3: Run Guided Queries ---

if __name__ == "__main__":

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
    (
            "Create a line chart showing average happiness_score by region "
            "for each year from 2015 to 2024. Use the actual dataset and "
            "save the chart to outputs/happiness_by_region.png."
    )
]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # --- Create Regional Happiness Chart Using Actual CSV Files ---

    frames = []

    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["region"] = year_df["region"].str.strip()
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    happiness_df = pd.concat(frames, ignore_index=True)

    avg_happiness = (
        happiness_df
        .groupby(["region", "year"])["happiness_score"]
        .mean()
        .reset_index()
    )

    print("Years included:", sorted(avg_happiness["year"].unique()))

    plt.figure(figsize=(12, 6))

    for region in sorted(avg_happiness["region"].unique()):
        region_data = avg_happiness[avg_happiness["region"] == region]
        plt.plot(
            region_data["year"],
            region_data["happiness_score"],
            marker="o",
            label=region
        )

    plt.title("Average Happiness Score by Region (2015-2024)")
    plt.xlabel("Year")
    plt.ylabel("Average Happiness Score")
    plt.xticks(range(2015, 2025))
    plt.legend(title="Region")
    plt.grid()
    plt.tight_layout()

    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/happiness_by_region.png")
    plt.close()

    print("Chart successfully created using the actual CSV files.")

    # --- Task 4: Your Own Questions ---

    # My query 1
    my_query_1 = "Show me the top 5 countries with the highest GDP per capita in 2023."
    response_1 = agent.run(my_query_1, reset=False)
    print("\n--- My Query 1 ---")
    print(response_1)

    # Comment: This triggered tool use because the agent used
    # get_top_n_countries to retrieve the top 5 countries.

    # My query 2
    my_query_2 = (
        "Use get_top_n_countries to get the top 5 happiest countries "
        "in 2015 and 2024. Then write Python code to compare both lists "
        "and identify which countries appear in both years."
    )
    response_2 = agent.run(my_query_2, reset=False)
    print("\n--- My Query 2 ---")
    print(response_2)

    # Comment: This triggered tool use and code generation.
    # The agent retrieved both rankings and used Python code to compare them.


# --- Task 5: Reflection ---
#
# 1. The agent reported a positive correlation (Pearson r = 0.6313)
#    between GDP per capita and happiness score. The p-value was 0.0,
#    indicating a statistically significant relationship. The agent
#    communicated the direction and strength of the relationship clearly.
#
# 2. I was surprised that the agent could use tools and write Python
#    code to compare the happiest countries in two different years.
#    It made the analysis easier and faster.
#
# 3. An additional tool that would be useful is a tool to compare
#    two countries' happiness scores across multiple years. It would
#    help answer questions about how individual countries' happiness
#    has changed over time.



