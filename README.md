# genai-training
Practicum environment and application  blueprint for GenAI Training

## Install
- Create environment and update
    ```
    poetry shell  
    poetry update
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