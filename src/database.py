# =============================================================
# FILE: src/database.py
# PURPOSE: SQLite database for storing fraud predictions
# =============================================================
#
# WHAT IS THIS FILE?
# ------------------
# In real banks, every fraud prediction must be STORED somewhere
# for auditing, compliance, and analytics. This file creates a
# SQLite database that saves every prediction the system makes.
#
# WHAT IS SQLite?
# ---------------
# SQLite is a lightweight database that stores everything in a
# single file (fraud_detection.db). No need to install MySQL or
# PostgreSQL. Python has SQLite built-in — no extra installation!
#
# WHY IS THIS IMPORTANT FOR INTERVIEWS?
# --------------------------------------
# It shows you understand:
# 1. Data persistence (not just in-memory predictions)
# 2. SQL queries (a must-have skill for data scientists)
# 3. Audit trails (required in financial systems)
# 4. Production-grade thinking (not just Jupyter notebooks)
#
# HOW TO RUN:
# python src/database.py
# (This will create the database and run a quick test)
# =============================================================

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

# sqlite3 comes built-in with Python — no pip install needed!
# It lets us create and interact with SQLite databases
import sqlite3

# json lets us convert Python dictionaries to JSON strings
# We use this to store SHAP feature values as text in the DB
import json

# datetime gives us the current date and time
# We timestamp every prediction for auditing purposes
from datetime import datetime

# os helps us work with file paths
# We use it to build the path to our database file
import os

# ---------------------------------------------------------
# DATABASE CLASS
# ---------------------------------------------------------
# We wrap everything in a class called FraudDatabase
# A class is like a blueprint — it groups related functions
# together. Think of it as a "database manager" object.
# ---------------------------------------------------------

class FraudDatabase:
    """
    Manages the SQLite database for fraud predictions.

    This class provides methods to:
    1. Create the database and table (if they don't exist)
    2. Insert new predictions
    3. Query prediction history
    4. Get summary statistics
    """

    def __init__(self, db_path=None):
        """
        Initialize the database connection.

        WHAT IS __init__?
        -----------------
        __init__ is a special Python method that runs
        automatically when you create an object from this class.

        Example:
            db = FraudDatabase()  # This runs __init__

        Parameters:
        -----------
        db_path : str, optional
            Path to the SQLite database file.
            If not provided, it creates 'fraud_detection.db'
            in the project root directory.
        """

        # If no path is given, use the default location
        if db_path is None:

            # os.path.dirname gets the parent directory
            # __file__ is the path of THIS Python file (database.py)
            # We go up one level (from src/ to project root)
            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )

            # Join the project root with our database filename
            db_path = os.path.join(
                base_dir,
                "fraud_detection.db"
            )

        # Store the path so other methods can use it
        self.db_path = db_path

        # Create the table when the object is created
        self._create_table()

    @staticmethod
    def _to_float(value):
        """
        Safely convert a value to a native Python float.

        SQLite does NOT accept numpy types (arrays, scalars).
        This method handles:
        - None → None
        - numpy array [5.2] → 5.2
        - numpy scalar → 5.2
        - list [5.2] → 5.2
        - Python float → 5.2
        """
        if value is None:
            return None
        # If it's an array-like (numpy array or list), grab first element
        if hasattr(value, '__len__') and not isinstance(value, str):
            value = value[0]
        return float(value)

    def _create_table(self):
        """
        Create the predictions table if it doesn't exist.

        WHAT IS A TABLE?
        ----------------
        A table is like an Excel spreadsheet inside the database.
        Each row = one prediction
        Each column = one piece of information about that prediction

        THE UNDERSCORE PREFIX (_create_table):
        In Python, a method starting with _ means "private" —
        it's meant to be used internally by the class, not
        called directly from outside.
        """

        # 'with' automatically handles opening and closing
        # the database connection. If anything goes wrong,
        # it safely closes the connection.
        with sqlite3.connect(self.db_path) as conn:

            # A cursor is like a pointer that executes SQL commands
            cursor = conn.cursor()

            # This SQL command creates our predictions table
            # IF NOT EXISTS means: only create if it doesn't
            # already exist (prevents errors on re-runs)
            cursor.execute("""

                CREATE TABLE IF NOT EXISTS predictions (

                    -- id: unique number for each prediction
                    -- AUTOINCREMENT means it automatically
                    -- increases (1, 2, 3, 4...)
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- timestamp: when the prediction was made
                    -- TEXT because we store it as a string
                    -- like "2025-06-27 04:30:00"
                    timestamp TEXT NOT NULL,

                    -- fraud_prob: the model's probability
                    -- REAL means decimal number (0.0 to 1.0)
                    fraud_prob REAL NOT NULL,

                    -- is_fraud: 1 = fraud, 0 = legitimate
                    -- INTEGER because it's either 0 or 1
                    is_fraud INTEGER NOT NULL,

                    -- risk_level: "low", "medium", "high", "critical"
                    risk_level TEXT NOT NULL,

                    -- top_features: JSON string of top SHAP features
                    -- Example: '{"V14": -0.52, "V10": -0.38}'
                    top_features TEXT,

                    -- ai_explanation: plain English explanation
                    -- from LangChain + Gemini
                    ai_explanation TEXT,

                    -- amount_log: log-transformed transaction amount
                    amount_log REAL,

                    -- hour: hour of day (0-23)
                    hour REAL
                )
            """)

            # Save the changes to the database file
            conn.commit()

    def insert_prediction(
        self,
        fraud_prob,
        is_fraud,
        risk_level,
        top_features=None,
        ai_explanation=None,
        amount_log=None,
        hour=None
    ):
        """
        Save a new prediction to the database.

        This is called every time the model makes a prediction.
        It creates a permanent record for auditing.

        Parameters:
        -----------
        fraud_prob : float
            The model's fraud probability (0.0 to 1.0)

        is_fraud : bool or int
            Whether the model flagged this as fraud

        risk_level : str
            "low", "medium", "high", or "critical"

        top_features : dict, optional
            Top SHAP features and their values
            Example: {"V14": -0.52, "V10": -0.38}

        ai_explanation : str, optional
            Plain English explanation from LangChain

        amount_log : float, optional
            Log-transformed transaction amount

        hour : float, optional
            Hour of the transaction (0-23)

        Returns:
        --------
        int : The id of the inserted prediction
        """

        with sqlite3.connect(self.db_path) as conn:

            cursor = conn.cursor()

            # SQL INSERT command — adds a new row to the table
            # The ? marks are placeholders — they get replaced
            # with actual values. This prevents SQL injection
            # (a security attack where hackers put malicious
            # SQL code in the input)
            cursor.execute("""

                INSERT INTO predictions (
                    timestamp,
                    fraud_prob,
                    is_fraud,
                    risk_level,
                    top_features,
                    ai_explanation,
                    amount_log,
                    hour
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            """, (
                # Current date and time as a string
                datetime.now().isoformat(),

                # Round to 6 decimal places
                round(float(fraud_prob), 6),

                # Convert boolean to integer (True → 1, False → 0)
                int(is_fraud),

                # Risk level as-is
                str(risk_level),

                # Convert dictionary to JSON string
                # json.dumps turns {"V14": -0.5} into '{"V14": -0.5}'
                json.dumps(top_features) if top_features else None,

                # AI explanation as-is
                ai_explanation,

                # Amount and hour — ensure native Python float
                # (SQLite rejects numpy arrays/scalars)
                self._to_float(amount_log),
                self._to_float(hour)
            ))

            conn.commit()

            # lastrowid gives us the id of the row we just inserted
            return cursor.lastrowid

    def get_recent_predictions(self, limit=50):
        """
        Get the most recent predictions.

        Parameters:
        -----------
        limit : int
            How many predictions to return (default: 50)

        Returns:
        --------
        list of dict : Each prediction as a dictionary
        """

        with sqlite3.connect(self.db_path) as conn:

            # row_factory = sqlite3.Row lets us access columns
            # by name instead of index number
            # Without it: row[0], row[1], row[2]
            # With it: row["timestamp"], row["fraud_prob"]
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()

            # SELECT * means "get all columns"
            # ORDER BY timestamp DESC means "newest first"
            # LIMIT ? means "only return this many rows"
            cursor.execute("""

                SELECT *
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?

            """, (limit,))

            # fetchall() gets all matching rows
            rows = cursor.fetchall()

            # Convert each row to a dictionary
            results = []

            for row in rows:

                result = dict(row)

                # Parse the JSON string back into a dictionary
                if result.get("top_features"):
                    result["top_features"] = json.loads(
                        result["top_features"]
                    )

                results.append(result)

            return results

    def get_predictions_by_risk(self, risk_level):
        """
        Get all predictions filtered by risk level.

        Example:
            db.get_predictions_by_risk("critical")
            # Returns all transactions flagged as critical risk

        Parameters:
        -----------
        risk_level : str
            "low", "medium", "high", or "critical"

        Returns:
        --------
        list of dict : Matching predictions
        """

        with sqlite3.connect(self.db_path) as conn:

            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()

            # WHERE risk_level = ? filters rows
            # Only returns rows where risk_level matches
            cursor.execute("""

                SELECT *
                FROM predictions
                WHERE risk_level = ?
                ORDER BY timestamp DESC

            """, (risk_level,))

            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_summary_stats(self):
        """
        Get summary statistics about all predictions.

        Returns a dictionary with:
        - total predictions
        - total fraud detected
        - fraud rate
        - count by risk level
        - average fraud probability

        This is used by the dashboard for the overview page.
        """

        with sqlite3.connect(self.db_path) as conn:

            cursor = conn.cursor()

            # COUNT(*) counts all rows
            cursor.execute(
                "SELECT COUNT(*) FROM predictions"
            )
            total = cursor.fetchone()[0]

            # If no predictions yet, return zeros
            if total == 0:
                return {
                    "total_predictions": 0,
                    "total_fraud": 0,
                    "fraud_rate": 0.0,
                    "avg_fraud_prob": 0.0,
                    "risk_counts": {
                        "low": 0,
                        "medium": 0,
                        "high": 0,
                        "critical": 0
                    }
                }

            # Count fraud predictions
            # SUM(is_fraud) adds up all the 1s
            cursor.execute(
                "SELECT SUM(is_fraud) FROM predictions"
            )
            total_fraud = cursor.fetchone()[0] or 0

            # Average fraud probability
            # AVG() calculates the average of a column
            cursor.execute(
                "SELECT AVG(fraud_prob) FROM predictions"
            )
            avg_prob = cursor.fetchone()[0] or 0.0

            # Count by risk level
            # GROUP BY groups rows with the same risk_level
            # COUNT(*) then counts each group
            cursor.execute("""

                SELECT risk_level, COUNT(*)
                FROM predictions
                GROUP BY risk_level

            """)

            risk_counts = {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }

            for row in cursor.fetchall():
                risk_counts[row[0]] = row[1]

            return {
                "total_predictions": total,
                "total_fraud": int(total_fraud),
                "fraud_rate": round(
                    total_fraud / total * 100, 2
                ),
                "avg_fraud_prob": round(avg_prob, 4),
                "risk_counts": risk_counts
            }

    def clear_all(self):
        """
        Delete all predictions from the database.

        WARNING: This cannot be undone!
        Used mainly during testing/development.
        """

        with sqlite3.connect(self.db_path) as conn:

            cursor = conn.cursor()

            cursor.execute("DELETE FROM predictions")

            conn.commit()

            print("All predictions cleared from database.")


# =============================================================
# MAIN — TEST THE DATABASE
# =============================================================
# This code only runs when you execute this file directly:
#   python src/database.py
#
# It does NOT run when another file imports this module:
#   from src.database import FraudDatabase
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING DATABASE MODULE")
    print("=" * 60)

    # Create a database instance
    db = FraudDatabase()

    print(f"\nDatabase location: {db.db_path}")

    # ---------------------------------------------------------
    # TEST 1: Insert sample predictions
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("TEST 1: Inserting sample predictions...")
    print("-" * 40)

    # Insert a fraud prediction
    id1 = db.insert_prediction(
        fraud_prob=0.92,
        is_fraud=True,
        risk_level="critical",
        top_features={
            "V14": -4.21,
            "V10": -3.85,
            "V12": -2.91
        },
        ai_explanation=(
            "This transaction is flagged as CRITICAL RISK "
            "with 92% fraud probability. The primary "
            "indicators are extreme negative values in "
            "V14 and V10, which are strong fraud signals."
        ),
        amount_log=6.8,
        hour=2.5
    )

    print(f"Inserted fraud prediction with id: {id1}")

    # Insert a legitimate prediction
    id2 = db.insert_prediction(
        fraud_prob=0.03,
        is_fraud=False,
        risk_level="low",
        top_features={
            "V14": 0.12,
            "V10": 0.34
        },
        ai_explanation=(
            "This transaction appears LEGITIMATE with only "
            "3% fraud probability. All feature values are "
            "within normal ranges."
        ),
        amount_log=3.2,
        hour=14.0
    )

    print(f"Inserted legit prediction with id: {id2}")

    # Insert a medium risk prediction
    id3 = db.insert_prediction(
        fraud_prob=0.45,
        is_fraud=False,
        risk_level="medium",
        top_features={
            "V14": -1.5,
            "V17": -0.8
        },
        ai_explanation=(
            "This transaction has MEDIUM RISK with 45% "
            "fraud probability. Some features show mild "
            "anomalies but are not conclusive."
        ),
        amount_log=5.1,
        hour=23.0
    )

    print(f"Inserted medium risk prediction with id: {id3}")

    # ---------------------------------------------------------
    # TEST 2: Retrieve recent predictions
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("TEST 2: Retrieving recent predictions...")
    print("-" * 40)

    recent = db.get_recent_predictions(limit=5)

    for pred in recent:
        print(
            f"  ID: {pred['id']} | "
            f"Prob: {pred['fraud_prob']:.2f} | "
            f"Risk: {pred['risk_level']} | "
            f"Time: {pred['timestamp'][:19]}"
        )

    # ---------------------------------------------------------
    # TEST 3: Filter by risk level
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("TEST 3: Filtering by risk level...")
    print("-" * 40)

    critical = db.get_predictions_by_risk("critical")

    print(f"  Critical risk predictions: {len(critical)}")

    # ---------------------------------------------------------
    # TEST 4: Summary statistics
    # ---------------------------------------------------------
    print("\n" + "-" * 40)
    print("TEST 4: Summary statistics...")
    print("-" * 40)

    stats = db.get_summary_stats()

    print(f"  Total predictions : {stats['total_predictions']}")
    print(f"  Total fraud       : {stats['total_fraud']}")
    print(f"  Fraud rate        : {stats['fraud_rate']}%")
    print(f"  Avg fraud prob    : {stats['avg_fraud_prob']}")
    print(f"  Risk breakdown    : {stats['risk_counts']}")

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL DATABASE TESTS PASSED!")
    print("=" * 60)

    print(f"\nDatabase file created at:")
    print(f"  {db.db_path}")

    print("\nYou can view it with any SQLite browser:")
    print("  https://sqlitebrowser.org/")
