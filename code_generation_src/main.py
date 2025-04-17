from LLM_langgraph import app

question = "How to write the code of a columnbic intercations ?"
solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": ""})