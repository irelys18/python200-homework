
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import os
import json
from scipy.stats import pearsonr
from smolagents import tool
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel



# Load the API key and create the OpenAI client.
load_dotenv()
client = OpenAI()


# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# Define the JSON schema that describes the function to an LLM.
# The schema tells the LLM the function name, purpose, and required input.

celsius_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in degrees Celsius."
            }
        },
        "required": ["celsius"]
    }
}

# Call the function directly with three test temperatures.

print("\n--- Q1: Celsius to Fahrenheit ---")

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# Display the JSON schema.

print("\nFunction JSON Schema:")
print(celsius_schema)


# Q2

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Describe the tool using a JSON schema.
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': celsius_schema,
    },
]

def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a helpful assistant.
Use get_current_time whenever the user asks for the current time.
Use celsius_to_fahrenheit whenever the user asks to convert
a Celsius temperature to Fahrenheit.
For other questions, answer using your own knowledge."""

    # Step 1: Start the conversation.
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: First API call. The model decides whether to use a tool.
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )

    print("First response received from model...")
    print(first_response)

    first_message = first_response.choices[0].message

    # Record the assistant's response.
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: Check whether the model requested a tool.
    if first_message.tool_calls:
        print("Agentic mode engaged...")

        # Go through each tool requested by the model.
        for tool_call in first_message.tool_calls:

            # Get the name of the requested function.
            function_name = tool_call.function.name

            if function_name == 'get_current_time':
                tool_result = get_current_time()

            elif function_name == 'celsius_to_fahrenheit':
                arguments = json.loads(tool_call.function.arguments)
                celsius = arguments['celsius']
                tool_result = celsius_to_fahrenheit(celsius)

            else:
                tool_result = f'Error: unknown tool {function_name}.'

            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Send the tool result back to the conversation.
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: Second API call after tool execution.
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )

        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''

    else:
        print("No tools needed....")

    # If no tool was requested, return the first response.
    return first_message.content or ''


# Prediction:
# 1. I predict that the agent will NOT call a tool because
#    the only available tool gets the current time, and the
#    user is asking for a temperature conversion.
#
# 2. I predict that only ONE API call will be made because
#    the model can answer directly without using a tool.

print("\n--- Q2: Testing the Agent ---")

answer = run_agent("Convert 100 degrees Celsius to Fahrenheit")

print("\nFinal Answer:")
print(answer)

# Check whether the prediction was correct.
# "No tools needed" means the model answered directly.
print("\nPrediction check:")
print("Expected: No tool calls and one API call.")


# Q3

print("\n--- Q3: Testing Both Tools ---")

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# The Celsius conversion tool should be called.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# No tool should be called because this is a general knowledge question.


# Q4
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []

        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"

            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }

        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()

        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }

        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error:
            return error

        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary statistics for selected columns or all columns."""
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]

            if missing:
                return {"error": f"These columns are not in the data: {missing}"}

            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Return descriptive statistics for one column."""
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": (
                    f"'{column}' is not a column. "
                    f"Options: {self.df.columns.tolist()}"
                )
            }

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    # --- Q4: New correlation method ----------------------------------

    def compute_correlation(self, col1: str, col2: str):
        """Calculate the Pearson correlation between two numeric columns."""

        # Check that a CSV file has been loaded.
        error = self._ensure_loaded()
        if error:
            return error

        # Check that both requested columns exist.
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {
                "error": (
                    "One or both columns are missing. "
                    f"Available columns: {self.df.columns.tolist()}"
                )
            }

        # Calculate Pearson correlation and p-value.
        pearson_r, p_value = pearsonr(self.df[col1], self.df[col2])

        # Return the results rounded to four decimal places.
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(pearson_r), 4),
            "p_value": round(float(p_value), 4),
        }

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot one or two columns from the active CSV."""

        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If x and y are the same, plot y against the row index.
        if x == y:
            x = None

        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()

            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."


print("Class defined")

# Create the CSV manager and point it to the resources folder.
RESOURCES_DIR = Path(__file__).parent / "resources"

csv_backend = CsvManager(RESOURCES_DIR)

csv_manager = csv_backend

# Connect the agent's tool names to the actual Python methods.
node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "compute_correlation": csv_backend.compute_correlation,
    "plot_data": csv_backend.plot_data,
}

# Describe the available tools so the AI knows how to use them.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename, such as bike_commute.csv.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show summary statistics for selected columns or all columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show summary statistics for one column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Calculate Pearson correlation and p-value between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "First numeric column name.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Second numeric column name.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string"},
                    "x": {"type": "string"},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                    },
                },
                "required": ["y"],
            },
        },
    },
]

print("CSV tools and schemas defined.")

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run one ReAct agent cycle using the available tools.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """Convert a tool result into a message for the model."""
        content = (
            json.dumps(result, default=str)
            if not isinstance(result, str)
            else result
        )
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: Ask the model what to do next.
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Save the assistant message.
        assistant_entry = {
            "role": "assistant",
            "content": msg.content,
        }

        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                tc.model_dump() for tc in msg.tool_calls
            ]

        messages.append(assistant_entry)

        # If there are no tool calls, return the final answer.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: Run the requested tools.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(
                tool_call.function.arguments or "{}"
            )

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)

            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(
                        f"Tool error in {name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    result = {
                        "error": (
                            f"Tool '{name}' failed: "
                            f"{type(e).__name__}: {e}"
                        )
                    }

            # Send the tool result back to the model.
            messages.append(
                observe_tool_result(tool_call.id, result)
            )

    return "I hit the tool-round limit. Try a simpler request."

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

def run_agent():
    """
    Simple command-line chat loop so it feels like a chatbot.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("CSV data agent at your service. Here to help look at your CSV data!")
    print("Type a question. Type 'exit' to quit.\n")
    print("To start, try 'list csv files' or 'load bike_commute.csv'\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        print(f"User query: {user_text}")
        assistant_text = run_agent_cycle(messages, user_text)
        print(f"\nAssistant: {assistant_text}\n")


# --- Q4: Test CSV Manager ---

print("\n--- Q4: Testing CSV Manager ---")

print("Available CSV files:")
print(csv_backend.list_csv_files())

print("\nLoading bike_commute.csv:")
print(csv_backend.load_csv("bike_commute.csv"))

print("\nCSV columns:")
print(csv_backend.get_columns())

print("\nTesting correlation:")
print(csv_backend.compute_correlation("distance_km", "duration_min"))

# --- Q5: Run the CSV data agent ---
# run_agent()
# --- Q5: Test agent with multiple tools ---

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print("\nQ5 Agent Final Response:")
print(result)

# --- Q6: Print the full conversation history ---

# system: Instructions that define the agent's behavior.
# user: The question or request from the user.
# assistant: The model's reasoning response or tool-call request.
# tool: The result returned by a Python function called by the assistant.

print("\nQ6: Full ReAct Conversation History:")
print(json.dumps(messages, indent=2, default=str))


# --- Q7: Wrap compute_correlation as a smolagents tool ---

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two numeric CSV columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation,
        and p-value.
    """
    return csv_manager.compute_correlation(col1, col2)


# smolagents automatically generates the tool description and parameter
# information from the function name, type hints, and docstring.
# Unlike Q4, we do not need to manually write a JSON schema.
# The developer must provide clear documentation, parameter descriptions,
# and accurate type hints so the generated tool description is useful.

print("\nQ7: smolagents Tool Description:")
print(compute_correlation.description)



# --- Q8: Define the CSV tools for smolagents ---

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dictionary containing available CSV filenames.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/.

    Args:
        filename: Name of the CSV file to load.

    Returns:
        A dictionary with loading status and column names.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names from the loaded CSV.

    Returns:
        A list of column names or an error dictionary.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary statistics for selected CSV columns.

    Args:
        columns: Optional list of column names to summarize.

    Returns:
        A dictionary containing summary statistics.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a column using basic statistics.

    Args:
        column: Name of the column to describe.

    Returns:
        A dictionary containing descriptive statistics.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(
    y: str,
    x: str | None = None,
    plot_type: str = "line",
) -> str | dict:
    """Plot data from the currently loaded CSV.

    Args:
        y: Column to plot on the y-axis.
        x: Column to plot on the x-axis.
        plot_type: Type of plot, either line or scatter.

    Returns:
        A message describing the plot or an error.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


# --- Q8: Collect all smolagents tools ---

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]


# --- Q8: Create both agents ---

model_to_use = "gpt-4o-mini"

model = OpenAIServerModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot"],
)

# --- Q8: Compare ToolCallingAgent and CodeAgent ---

prompt = (
    "Load bike_commute.csv using csv_manager. "
    "Use Python code to create a scatter plot. "
    "Set duration_min on the x-axis and avg_heart_rate on the y-axis. "
    "The dots must be green. "
    "Use matplotlib.pyplot and csv_manager.df. "
    "Do not calculate correlation. Execute the plotting code."
)

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager},
)

print("\nQ8: ToolCallingAgent Response:")
print(response_tool)

print("\nQ8: CodeAgent Response:")
print(response_code)


# 1. What did each agent produce?
# ToolCallingAgent generated Python code for a green scatter 
# plot but did not execute it. CodeAgent prepared the data 
# and returned a code snippet for plotting, but did not display 
# the graph.

# 2. When is each agent more useful?
# ToolCallingAgent is useful for tasks supported by predefined 
# tools. CodeAgent is more flexible because it can generate custom 
# Python code for more complex tasks, such as customizing plots.


# Q9: Reflection 

# 1. A ToolCallingAgent is a better choice for tasks like loading a CSV
#    or calculating a correlation using predefined functions. These tasks
#    have clear steps and require controlled, predictable actions.

# 2. A CodeAgent can generate and execute its own Python code. This creates
#    a security risk because generated code could perform unintended actions,
#    modify files, or access information. A ToolCallingAgent is more limited
#    to the functions explicitly provided to it.
