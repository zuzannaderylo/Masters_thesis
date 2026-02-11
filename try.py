"""
try.py

Additional script used for pipeline execution.
"""


from scripts.run_pipeline import run_pipeline

copenhagen_score = run_pipeline("Copenhagen")

gdansk_score = run_pipeline("Gdansk")