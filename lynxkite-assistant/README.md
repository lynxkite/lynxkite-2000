# Lynxkite Assistant

AI assistant backend for LynxKite.

This is a separate optional package to avoid burdening everyone with its dependencies.

## Setup
To use the assistant, you will need to specify a language model, and the required API key for that model.
You can use any model [supported](https://docs.langchain.com/oss/python/deepagents/models) by the `deepagents` python library.
You will need to set the chosen model as an environment variable with `export LYNXKITE_ASSISTANT_MODEL='<model name>'`.

To ensure that the assistant works even after restarting your environment, you can add the following to `.venv/bin/activate`:
```
export LYNXKITE_ASSISTANT_MODEL='openai:gpt-5.4-mini'
export OPENAI_API_KEY=<api key>
```
Note: if you're not using an OpenAI model, the variable name for the API key will be different.

## Capabilities
The LynxKite Assistant can:
- provide information about the boxes
- add, delete and modify nodes in the workspace
- create custom boxes for specialized tasks
- add comments ot the workspace
- reorganize the boxes in the workspace
- with the proper setup, the assistant can access the internet and can make web searches and read websites

In your messages you can also reference the boxes you have selected with a click of a button, so the Assistant knows which boxes you're talking about.

## Internet access setup
If you want the agent to be able to access the internet you will need a service that provides this feature to the agent.
The agent's tools communicate via FireCrawl's API, any service that uses this should work. Here are some recommended options.
### Hosted
You can choose to use a hosted service: these require minimal setup but only give access to a limited amount of searches for free.
- [FireCrawl](https://www.firecrawl.dev/): requires API key, API access through <https://api.firecrawl.dev>
- [fastCRW](https://fastcrw.com/): requires API key, API access through <https://api.fastcrw.com>

After obtaining the API key, you have to set the following environmental variables:
```
export LYNXKITE_WEB_ACCESS_URL=<address of hosted api>
export LYNXKITE_WEB_ACCESS_API_KEY=<api key>
```
### Self-hosted
You can also host the service on your own machine. fastCRW has a docker container you can install and host easily with the following commands:
```
git clone https://github.com/us/crw.git
cd crw
docker compose up
```
After setting up the service, you need to set the `LYNXKITE_WEB_ACCESS_URL` environmental variable to the correct URL, usually `http://localhost:3000`.
