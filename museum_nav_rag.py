from typing import List, Dict, Optional, Tuple
import os
import tempfile
from datetime import datetime
import uuid
import asyncio
import base64
from io import BytesIO
from urllib.parse import urlparse

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import torch
from transformers import CLIPProcessor, CLIPModel
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

load_dotenv()

# Constants
COLLECTION_NAME = "museum-artifacts"
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

def init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "openai_api_key": "",
        "setup_complete": False,
        "client": None,
        "clip_model": None,
        "clip_processor": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral",
        "processed_images": [],
        "museum_name": "Alexander Black House",
        "current_location": "Main Floor"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar() -> None:
    """Configure sidebar with API settings and museum options."""
    with st.sidebar:
        st.title("🏛️ Museum Configuration")
        st.markdown("---")
        
        # API Configuration
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            type="password"
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password"
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password"
        )
        
        st.markdown("---")
        
        # Museum Settings
        st.markdown("### 🏛️ Museum Settings")
        st.session_state.museum_name = st.text_input(
            "Museum Name",
            value=st.session_state.museum_name
        )
        st.session_state.current_location = st.text_input(
            "Current Location",
            value=st.session_state.current_location
        )
        
        st.markdown("---")
        
        # Voice Settings
        st.markdown("### 🎤 Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
            help="Choose the voice for the audio response"
        )

def setup_qdrant_and_clip() -> Tuple[QdrantClient, CLIPModel, CLIPProcessor]:
    """Initialize Qdrant client and CLIP model for image embeddings."""
    if not all([st.session_state.qdrant_url, st.session_state.qdrant_api_key]):
        raise ValueError("Qdrant credentials not provided")
    
    # Parse and construct the base URL for Qdrant
    raw_url = st.session_state.qdrant_url
    if not raw_url.startswith(('http://', 'https://')):
        raw_url = 'https://' + raw_url
        
    parsed_url = urlparse(raw_url)
    qdrant_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
    # Initialize Qdrant client
    client = QdrantClient(
        url=qdrant_url,
        api_key=st.session_state.qdrant_api_key
    )
    
    # Initialize CLIP model for image embeddings
    st.info("Loading CLIP model for image processing...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test embedding to get dimensions
    test_image = Image.new('RGB', (224, 224), color='white')
    inputs = processor(images=test_image, return_tensors="pt")
    with torch.no_grad():
        test_embedding = model.get_image_features(**inputs)
    embedding_dim = test_embedding.shape[1]
    
    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
        st.success(f"Created collection '{COLLECTION_NAME}' with dimension {embedding_dim}")
    except Exception as e:
        if "already exists" not in str(e):
            raise e
        st.info(f"Using existing collection '{COLLECTION_NAME}'")
    
    return client, model, processor

def process_museum_images(uploaded_files) -> List[Dict]:
    """Process uploaded museum artifact images and extract metadata."""
    processed_images = []
    
    for uploaded_file in uploaded_files:
        try:
            # Load and process image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Extract filename without extension for artifact name
            artifact_name = os.path.splitext(uploaded_file.name)[0].replace('_', ' ').replace('-', ' ').title()
            
            # Create metadata
            metadata = {
                "artifact_name": artifact_name,
                "file_name": uploaded_file.name,
                "museum": st.session_state.museum_name,
                "location": st.session_state.current_location,
                "upload_timestamp": datetime.now().isoformat(),
                "image_size": f"{image.width}x{image.height}",
                "description": f"Museum artifact: {artifact_name} from {st.session_state.museum_name}"
            }
            
            processed_images.append({
                "image": image,
                "metadata": metadata
            })
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return processed_images

def generate_image_embedding(image: Image.Image, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    """Generate CLIP embedding for an image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.numpy().flatten()

def store_image_embeddings(
    client: QdrantClient,
    model: CLIPModel,
    processor: CLIPProcessor,
    processed_images: List[Dict],
    collection_name: str
) -> None:
    """Store image embeddings in Qdrant."""
    for img_data in processed_images:
        try:
            # Generate embedding
            embedding = generate_image_embedding(img_data["image"], model, processor)
            
            # Store in Qdrant
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload=img_data["metadata"]
                    )
                ]
            )
            st.success(f"✅ Stored: {img_data['metadata']['artifact_name']}")
            
        except Exception as e:
            st.error(f"Error storing {img_data['metadata']['artifact_name']}: {str(e)}")

def capture_camera_image() -> Optional[Image.Image]:
    """Capture image from camera using OpenCV."""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access camera")
            return None
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            st.error("Failed to capture image")
            return None
            
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def find_similar_artifacts(
    query_image: Image.Image,
    client: QdrantClient,
    model: CLIPModel,
    processor: CLIPProcessor,
    collection_name: str,
    limit: int = 3
) -> List[Dict]:
    """Find similar artifacts in the museum database."""
    try:
        # Generate embedding for query image
        query_embedding = generate_image_embedding(query_image, model, processor)
        
        # Search in Qdrant
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True
        )
        
        results = []
        for point in search_response.points:
            results.append({
                "score": point.score,
                "metadata": point.payload
            })
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

async def generate_artifact_description(
    artifact_metadata: Dict,
    similarity_score: float,
    openai_api_key: str,
    voice: str
) -> Dict:
    """Generate contextual description of the artifact using OpenAI."""
    try:
        # Create contextual prompt
        prompt = f"""
        You are a knowledgeable museum guide helping a visually impaired visitor. 
        
        The visitor is looking at: {artifact_metadata.get('artifact_name', 'Unknown artifact')}
        Location: {artifact_metadata.get('location', 'Unknown location')} in {artifact_metadata.get('museum', 'the museum')}
        Match confidence: {similarity_score:.2%}
        
        Provide a warm, engaging description that includes:
        1. What the artifact is and its significance
        2. Historical context and interesting details
        3. Physical description (materials, size, craftsmanship)
        4. Its role in the museum's collection
        
        Keep the description conversational and accessible, as if speaking directly to the visitor.
        Limit to 2-3 paragraphs for audio clarity.
        """
        
        # Generate text description
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        
        text_response = await async_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly, knowledgeable museum guide."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        description_text = text_response.choices[0].message.content
        
        # Generate audio
        audio_response = await async_openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=description_text,
            response_format="mp3"
        )
        
        # Save audio to temporary file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"artifact_description_{uuid.uuid4()}.mp3")
        
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)
        
        return {
            "status": "success",
            "text_description": description_text,
            "audio_path": audio_path,
            "artifact_name": artifact_metadata.get('artifact_name', 'Unknown artifact'),
            "similarity_score": similarity_score
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title="AI Museum Navigator",
        page_icon="🏛️",
        layout="wide"
    )
    
    init_session_state()
    setup_sidebar()
    
    st.title("🏛️ AI Museum Navigator")
    st.info("Upload museum artifact images to build your database, then use your camera to identify and learn about artifacts through voice descriptions!")
    
    # Display current museum info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Museum", st.session_state.museum_name)
    with col2:
        st.metric("Current Location", st.session_state.current_location)
    
    # Setup section
    if not st.session_state.setup_complete:
        st.header("🔧 Setup Museum Database")
        
        # Image upload section
        uploaded_files = st.file_uploader(
            "Upload Museum Artifact Images",
            type=SUPPORTED_IMAGE_FORMATS,
            accept_multiple_files=True,
            help="Upload images of museum artifacts to build the recognition database"
        )
        
        if uploaded_files:
            if st.button("Process and Store Images", type="primary"):
                with st.spinner('Setting up AI models and processing images...'):
                    try:
                        # Setup Qdrant and CLIP
                        if not all([st.session_state.client, st.session_state.clip_model, st.session_state.clip_processor]):
                            client, model, processor = setup_qdrant_and_clip()
                            st.session_state.client = client
                            st.session_state.clip_model = model
                            st.session_state.clip_processor = processor
                        
                        # Process images
                        processed_images = process_museum_images(uploaded_files)
                        
                        if processed_images:
                            # Store embeddings
                            store_image_embeddings(
                                st.session_state.client,
                                st.session_state.clip_model,
                                st.session_state.clip_processor,
                                processed_images,
                                COLLECTION_NAME
                            )
                            
                            # Update processed images list
                            new_artifacts = [img["metadata"]["artifact_name"] for img in processed_images]
                            st.session_state.processed_images.extend(new_artifacts)
                            st.session_state.setup_complete = True
                            
                            st.success(f"✅ Successfully processed {len(processed_images)} artifacts!")
                            st.balloons()
                    
                    except Exception as e:
                        st.error(f"Setup error: {str(e)}")
    
    # Display processed artifacts
    if st.session_state.processed_images:
        st.sidebar.header("📚 Museum Database")
        st.sidebar.write(f"**{len(st.session_state.processed_images)} artifacts stored:**")
        for artifact in st.session_state.processed_images:
            st.sidebar.text(f"🏺 {artifact}")
    
    # Main interaction section
    if st.session_state.setup_complete:
        st.header("📸 Artifact Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Camera Capture")
            if st.button("📷 Take Photo of Artifact", type="primary", help="Capture an image to identify the artifact"):
                with st.spinner("Accessing camera..."):
                    captured_image = capture_camera_image()
                    
                    if captured_image:
                        st.image(captured_image, caption="Captured Image", use_column_width=True)
                        st.session_state.captured_image = captured_image
        
        with col2:
            st.subheader("Upload Image")
            uploaded_query = st.file_uploader(
                "Or upload an image to identify",
                type=SUPPORTED_IMAGE_FORMATS,
                help="Upload a photo of an artifact to identify it"
            )
            
            if uploaded_query:
                query_image = Image.open(uploaded_query).convert('RGB')
                st.image(query_image, caption="Uploaded Image", use_column_width=True)
                st.session_state.captured_image = query_image
        
        # Process captured/uploaded image
        if hasattr(st.session_state, 'captured_image') and st.session_state.captured_image:
            if st.button("🔍 Identify Artifact", type="secondary"):
                with st.status("Analyzing artifact...", expanded=True) as status:
                    try:
                        st.write("🔄 Searching museum database...")
                        
                        # Find similar artifacts
                        similar_artifacts = find_similar_artifacts(
                            st.session_state.captured_image,
                            st.session_state.client,
                            st.session_state.clip_model,
                            st.session_state.clip_processor,
                            COLLECTION_NAME
                        )
                        
                        if similar_artifacts:
                            best_match = similar_artifacts[0]
                            st.write(f"🎯 Found match: {best_match['metadata']['artifact_name']} ({best_match['score']:.2%} confidence)")
                            
                            st.write("🔄 Generating description...")
                            
                            # Generate description
                            result = asyncio.run(generate_artifact_description(
                                best_match['metadata'],
                                best_match['score'],
                                st.session_state.openai_api_key.strip(),
                                st.session_state.selected_voice
                            ))
                            
                            if result["status"] == "success":
                                status.update(label="✅ Artifact identified!", state="complete")
                                
                                # Display results
                                st.success(f"🏺 **{result['artifact_name']}**")
                                st.info(f"**Confidence:** {result['similarity_score']:.1%}")
                                
                                st.markdown("### 📝 Description:")
                                st.write(result["text_description"])
                                
                                st.markdown(f"### 🔊 Audio Description (Voice: {st.session_state.selected_voice})")
                                st.audio(result["audio_path"], format="audio/mp3")
                                
                                # Download button
                                with open(result["audio_path"], "rb") as audio_file:
                                    st.download_button(
                                        label="📥 Download Audio Description",
                                        data=audio_file.read(),
                                        file_name=f"{result['artifact_name'].replace(' ', '_')}_description.mp3",
                                        mime="audio/mp3"
                                    )
                                
                                # Show other similar artifacts
                                if len(similar_artifacts) > 1:
                                    st.markdown("### 🔍 Other Similar Artifacts:")
                                    for i, artifact in enumerate(similar_artifacts[1:], 1):
                                        st.write(f"{i+1}. {artifact['metadata']['artifact_name']} ({artifact['score']:.1%} match)")
                            
                            else:
                                status.update(label="❌ Error generating description", state="error")
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
                        else:
                            status.update(label="❌ No matches found", state="error")
                            st.warning("No similar artifacts found in the database. Try uploading more reference images or check the image quality.")
                    
                    except Exception as e:
                        status.update(label="❌ Error during analysis", state="error")
                        st.error(f"Analysis error: {str(e)}")
    
    else:
        st.info("👆 Please upload museum artifact images first to set up the recognition database!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🎓 **AI Museum Navigator** - Developed for accessible museum experiences\n\n"
        "Powered by CLIP image recognition, Qdrant vector database, and OpenAI voice synthesis"
    )

if __name__ == "__main__":
    main()