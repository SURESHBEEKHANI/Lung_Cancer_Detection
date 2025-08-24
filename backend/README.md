# Lung Cancer Detection

This application performs real-time lung cancer detection using a YOLO-Seg model. It processes CT scans and returns segmented images highlighting potential lung cancer regions.

## Features
- Real-time lung cancer segmentation
- Intuitive UI built with Streamlit
- Easy image upload and visualization

## Setup & Installation
- Install Python 3.x.
- Install the required packages:
  - streamlit
  - ultralytics
  - Pillow
  - torchvision
- Clone/download this repository.
- Place the model weight file `Lung Cancer Detection.pt` in the `weights/` directory.
- Place your logo in the `logo/` directory.

## Running the Application
To run the app, execute the following command in your terminal:

```
streamlit run app.py
```

## Usage
1. Open the app in your browser.
2. Upload a CT image via the sidebar.
3. Click on the "🔍 Predict Lung Cancer" button.
4. View the segmented lung cancer output.

## License
This project is provided for educational purposes.

## API & Authentication (Quick Guide)

This backend exposes REST endpoints used by the frontend. Authentication uses JWT in the Authorization header.

- POST /auth/signup  -> body: { name, email, password }  -> returns { access_token, token_type, user }
- POST /auth/login   -> body: { email, password }       -> returns { access_token, token_type, user }

Protected endpoints require the header:
  Authorization: Bearer <access_token>

Example: to generate a report:
  POST /report/generate (multipart/form-data: file)
  Header: Authorization: Bearer <token>

If a request returns 401, re-authenticate using /auth/login.
