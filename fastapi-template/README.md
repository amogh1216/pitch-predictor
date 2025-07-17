# FastAPI Template

This project is a FastAPI application that provides a simple API for managing messages. It includes routes for adding messages and listing all messages, as well as an about page.

## Project Structure

```
fastapi-template
├── app
│   ├── main.py        # Contains the FastAPI application and routes
│   ├── models.py      # Defines the MsgPayload class for message structure
│   └── __init__.py    # Marks the app directory as a Python package
├── requirements.txt    # Lists the dependencies for the application
├── Dockerfile           # Instructions to build a Docker image for the application
└── README.md            # Documentation for the project
```

## Requirements

To run this application, you need to have the following dependencies installed:

- FastAPI
- Uvicorn (for serving the application)

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Running the Application

To run the FastAPI application, you can use Uvicorn. Execute the following command from the root of the project:

```
uvicorn app.main:app --reload
```

This will start the server at `http://127.0.0.1:8000`. You can access the API documentation at `http://127.0.0.1:8000/docs`.

## Docker

To build and run the application using Docker, follow these steps:

1. Build the Docker image:

   ```
   docker build -t fastapi-template .
   ```

2. Run the Docker container:

   ```
   docker run -d -p 8000:8000 fastapi-template
   ```

The application will be accessible at `http://localhost:8000`.

## About

This application serves as a template for building FastAPI applications and can be extended with additional features as needed.