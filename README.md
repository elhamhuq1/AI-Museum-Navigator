
# AI Museum Navigator

An innovative AI-powered system that transforms museum experiences for individuals with visual impairments. This project combines state-of-the-art computer vision, natural language processing, and audio synthesis to create an autonomous, immersive cultural heritage experience.

## Project Overview

Developed as a case study for the Alexander Black House, this system enables visually impaired visitors to independently explore and learn about museum artifacts through real-time image recognition and contextual audio descriptions.

## Key Features

-  **Real-time Artifact Identification**: Uses CLIP embeddings for zero-shot image recognition
-  **Voice-Activated Audio Descriptions**: OpenAI TTS integration with multiple voice options
-  **Contextual Museum Information**: GPT-3.5 powered descriptions with historical context
-  **Interactive Web Interface**: Streamlit-based UI for testing and development
-  **Vector Similarity Search**: Qdrant database for fast artifact matching
-  **Camera Integration**: Real-time image capture and processing
-  **Accessibility Focus**: Designed specifically for visually impaired users

##  Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Qdrant Cloud account and API key

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/elhamhuq1/AI-Museum-Navigator.git
   cd ai-tourism
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API keys:**

 - Get a Qdrant API key and URL from: https://qdrant.to/cloud
 - Get a OpenAI API key from: https://platform.openai.com/api-keys

##  Usage

### Running the Application

1. **Start the Streamlit app**

   ```bash
   streamlit run museum_nav_rag.py
   ```

2. **Configure API Settings**

   - Enter your Qdrant URL and API key in the sidebar
   - Enter your OpenAI API key
   - Set your museum name and current location

3. **Upload Museum Artifacts**

   - Use the file uploader to add images of museum artifacts
   - Click "Process and Store Images" to build the database
   - Wait for the CLIP model to load and process images

4. **Identify Artifacts**
   - Use the camera capture or upload an image
   - Click "Identify Artifact" to find matches
   - Listen to the generated audio description

##  How It Works

1. **Artifact Database Setup**: Museum staff upload images of artifacts and their metadata to the system. Each image is processed using CLIP to generate a unique vector embedding, which is stored in a Qdrant vector database.

2. **User Interaction**: A visually impaired visitor wears the custom device (or uses the web interface) and can verbally request information about their surroundings.

3. **Image Capture**: The device's camera (or the user's upload) captures an image of the environment or artifact.

4. **Artifact Recognition**: The system processes the image, generates a CLIP embedding, and searches the Qdrant database for the most similar artifact vectors using cosine similarity.

5. **Contextual Description Generation**: Once a match is found, the system uses OpenAI GPT to generate a contextualized description of the artifact, including historical context and physical details.

6. **Audio Playback**: The description is converted to natural-sounding speech using OpenAI's TTS and played back to the user, providing an immersive, accessible museum experience.
