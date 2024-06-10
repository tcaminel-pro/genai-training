# genai-training
Practicum environment and application  blueprint for GenAI Training

## Pre-requisite
- You should have API keys you plan to use in environment variable. <br>
    ex:
    ```
    export EDENAI_API_KEY='eyJhbGc.....'
    export AZURE_OPENAI_API_KEY="18f5ac....'
    ```

## Install
- Create environment and update
    ```
    poetry shell  
    poetry update
    export PYTHONPATH=":./python"
    ``` 
- edit app_conf.yaml 
    - Change default llm and embeddings model if needed
- Tests:
    - Test with CLI:
        ```
        python python/main_cli.py run "joke" 
        ```
    - Test with Streamlit webapp
        ``` 
        make webapp
        ```
        select "Runnable Playground", then "joke"
    - Test  with FastAPI
        ``` 
        make fast_api
        ```
        enter http://127.0.0.1:8000/docs    

    - Test  with LangServe : TODO