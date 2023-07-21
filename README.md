# BRIO_x_Alkemy
Repository for the data product imagined for the collaboration with BRIO

# Usage
To run the frontend locally:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/BRIO_x_Alkemy"
cd frontend
flask run --host=0.0.0.0 --debug
```
each endpoint (implemented and listed in ```frontend/app.py``` by ```@app.route('/endpoint')```) is rendered by templates (located in ```frontend/templates```). Modifying the templates will dynamically update the html shown in each page (you just need a refresh).
