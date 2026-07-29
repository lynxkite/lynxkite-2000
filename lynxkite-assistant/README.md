# Lynxkite Assistant

AI assistant backend for LynxKite.

This is a separate optional package to avoid burdening everyone with its dependencies.

## Setup
To use the assistant, you will need to specify a language model, and the required API key for that model.
You can use any model [supported](https://docs.langchain.com/oss/python/deepagents/models) by the `deepagents` python library.
You will need to set the choosen model as an environment variable with `export LYNXKITE_ASSISTANT_MODEL='<model name>'`.

To ensure that the assistant works even after restarting your environment, you can add the following to `.venv/bin/activate`:
```
export LYNXKITE_ASSISTANT_MODEL='openai:gpt-5.4-mini'
export OPENAI_API_KEY=<api key>
```
Note: if you're not using an OpenAI model, the variable name for the API key will be different.

## Capabilites
The LynxKite Assistant can:
- provide information about the boxes
- add, delete and modify nodes in the workspace
- create custom boxes for specialized tasks
- add comments ot the workspace
- reorganize the boxes in the workspace

In your messages you can also reference the boxes you have selected with a click of a button, so the Assistant knows which boxes you're talking about.
