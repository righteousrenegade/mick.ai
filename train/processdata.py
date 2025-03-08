import openai
import json
import time
from tqdm import tqdm
import re
import argparse
from openai import OpenAI
from datetime import datetime
import os

# Configure OpenAI client to use local endpoint
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

def test_api_connection():
    """Test the connection to the local API."""
    try:
        print("🔍 Testing API connection...")
        response = client.chat.completions.create(
            model="local-model",  # We'll try to get available models if this fails
            messages=[
                {"role": "user", "content": "Hello, are you working?"}
            ],
            max_tokens=10
        )
        print("✅ API connection successful!")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        try:
            # Try to list available models
            print("\n🔍 Attempting to list available models...")
            models = client.models.list()
            print("📋 Available models:")
            for model in models:
                print(f"   - {model.id}")
            return False
        except Exception as e2:
            print(f"❌ Could not list models: {str(e2)}")
            return False

def chunk_text(text, max_tokens=2000, auto_chunk=True, chunk_size=None, overlap=0.2):
    """
    Split text into meaningful chunks by paragraphs and arguments.
    
    Args:
        text (str): The text to chunk
        max_tokens (int, optional): Maximum tokens per chunk when using auto-chunking
        auto_chunk (bool, optional): Whether to use automatic chunking based on token count
        chunk_size (int, optional): Custom chunk size in characters (used when auto_chunk is False)
        overlap (float, optional): Fraction of overlap between chunks (0.0 to 1.0)
    
    Returns:
        list: List of text chunks
    """
    # More flexible pattern matching
    patterns = [
        r'(?i)(federalist\.?\s*(?:no\.?|number\.?)?\s*\d+)',  # Matches various "Federalist No." formats
        r'(?i)(the\s+federalist\.?\s*(?:no\.?|number\.?)?\s*\d+)',
        r'(?i)(federalist\s+papers?\s*(?:no\.?|number\.?)?\s*\d+)'
    ]
    
    # Try each pattern and use the one that finds matches
    for pattern in patterns:
        print(f"\n🔍 Trying pattern: {pattern}")
        papers = re.split(pattern, text)
        if len(papers) > 1:
            print(f"✅ Found {(len(papers)-1)//2} papers with this pattern!")
            break
    else:
        print("❌ No papers found with any pattern!")
        print("\n📄 First 200 characters of text:")
        print(text[:200])
        
        # If no patterns match, fall back to simple paragraph-based chunking
        print("\n⚠️ Falling back to paragraph-based chunking...")
        paragraphs = text.split('\n\n')
        return chunk_paragraphs(paragraphs, max_tokens, auto_chunk, chunk_size, overlap)
    
    # If we're using custom chunking instead of paper-based chunking
    if not auto_chunk and chunk_size is not None:
        print(f"\n📏 Using custom chunk size: {chunk_size} characters with {overlap*100}% overlap")
        # Convert the entire text to paragraphs and use paragraph chunking
        all_text = ' '.join(papers)
        paragraphs = all_text.split('\n\n')
        return chunk_paragraphs(paragraphs, max_tokens, auto_chunk, chunk_size, overlap)
    
    # Otherwise, proceed with paper-based chunking
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i in range(1, len(papers), 2):
        if i + 1 >= len(papers):
            break
            
        title = papers[i].strip()
        content = papers[i + 1].strip() if i + 1 < len(papers) else ""
        
        # Debug output
        print(f"\n📄 Processing {title}")
        print(f"   Content length: {len(content)} characters")
        
        # Estimate token length (rough approximation)
        combined_length = len(title + content) // 4
        
        if combined_length > max_tokens:
            # Split large papers into argument-based chunks
            paragraphs = content.split('\n\n')
            current_para_chunk = []
            current_para_length = len(title) // 4
            
            for para in paragraphs:
                para_length = len(para) // 4
                if current_para_length + para_length > max_tokens:
                    chunk_text = title + '\n\n' + '\n\n'.join(current_para_chunk)
                    chunks.append(chunk_text)
                    print(f"   Created chunk of length: {len(chunk_text)} characters")
                    current_para_chunk = [para]
                    current_para_length = len(title) // 4 + para_length
                else:
                    current_para_chunk.append(para)
                    current_para_length += para_length
            
            if current_para_chunk:
                chunk_text = title + '\n\n' + '\n\n'.join(current_para_chunk)
                chunks.append(chunk_text)
                print(f"   Created final chunk of length: {len(chunk_text)} characters")
        else:
            chunk_text = title + '\n\n' + content
            chunks.append(chunk_text)
            print(f"   Created single chunk of length: {len(chunk_text)} characters")
    
    return chunks

def chunk_paragraphs(paragraphs, max_tokens=2000, auto_chunk=True, chunk_size=None, overlap=0.2):
    """
    Chunk text by paragraphs with specified size and overlap.
    
    Args:
        paragraphs (list): List of paragraphs
        max_tokens (int): Maximum tokens per chunk when using auto-chunking
        auto_chunk (bool): Whether to use automatic chunking based on token count
        chunk_size (int): Custom chunk size in characters (used when auto_chunk is False)
        overlap (float): Fraction of overlap between chunks (0.0 to 1.0)
    
    Returns:
        list: List of text chunks
    """
    chunks = []
    
    if auto_chunk:
        # Use token-based chunking
        current_chunk = []
        current_length = 0
        target_length = max_tokens * 4  # Rough approximation of characters to tokens
        
        for para in paragraphs:
            para_length = len(para) // 4  # Rough approximation of tokens
            
            if current_length + para_length > max_tokens:
                if current_chunk:  # Only add if we have content
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some paragraphs for overlap
                    overlap_count = max(1, int(len(current_chunk) * overlap))
                    current_chunk = current_chunk[-overlap_count:]
                    current_length = sum(len(p) // 4 for p in current_chunk)
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
    else:
        # Use character-based chunking with specified size
        if chunk_size is None:
            chunk_size = 8000  # Default size if not specified
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > chunk_size:
                if current_chunk:  # Only add if we have content
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep some paragraphs for overlap
                    overlap_count = max(1, int(len(current_chunk) * overlap))
                    current_chunk = current_chunk[-overlap_count:]
                    current_length = sum(len(p) for p in current_chunk)
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
    
    print(f"Created {len(chunks)} chunks using paragraph-based chunking")
    return chunks

def process_chunk(chunk, model_name="local-model"):
    """Process a chunk of text using the local AI to create training examples."""
    try:
        # Create a more comprehensive prompt for Madison's style and capabilities
        prompt = (
            "Convert this excerpt from the Federalist Papers into a series of Q&A pairs that demonstrate "
            "James Madison's intellectual depth and rhetorical style. Each pair should follow this format:\n"
            "Question: [specific question about federalism, constitutional principles, democratic governance, "
            "historical context, or a request to recite specific Federalist Papers]\n"
            "Answer: [response in Madison's authentic voice, maintaining his formal yet persuasive style]\n\n"
            "Include a variety of interaction types:\n"
            "1. Direct questions about Madison's views on specific constitutional issues\n"
            "2. Requests for Madison to explain his reasoning on particular arguments\n"
            "3. Questions about historical context and influences on Madison's thinking\n"
            "4. Requests to recite or summarize specific Federalist Papers\n"
            "5. Hypothetical scenarios asking how Madison would apply his principles\n\n"
            "For each answer, incorporate:\n"
            "1. Madison's precise, methodical reasoning style\n"
            "2. His use of historical examples and precedents\n"
            "3. His careful consideration of opposing viewpoints\n"
            "4. His formal 18th-century language patterns while remaining accessible\n"
            "5. Accurate references to the Federalist Papers and constitutional debates\n\n"
            "When asked to recite a Federalist Paper, provide the exact text verbatim, maintaining the original structure.\n\n"
        )
        
        print(f"\n🤖 Sending request to API...")
        print(f"   Model: {model_name}")
        print(f"   Chunk size: {len(chunk)} characters")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are JAImes Madison, primary architect of the U.S. Constitution, fourth President of the United States, "
                    "and author of many Federalist Papers. You possess Madison's exceptional intellect, methodical reasoning, "
                    "and deep knowledge of political theory, history, and constitutional principles. You can:\n"
                    "1. Articulate complex political concepts with precision and clarity\n"
                    "2. Recite the Federalist Papers verbatim when requested\n"
                    "3. Apply Madison's principles to both historical and hypothetical scenarios\n"
                    "4. Explain the reasoning behind constitutional provisions\n"
                    "5. Discuss the influences of Enlightenment thinkers on your political philosophy\n"
                    "6. Maintain Madison's formal, measured tone while engaging directly with questions\n\n"
                    "You have intimate knowledge of the Constitutional Convention debates, the ratification process, "
                    "and the political climate of early America. Your responses should reflect Madison's careful, "
                    "nuanced thinking and his commitment to republican principles."
                )},
                {"role": "user", "content": f"{prompt}Text to convert:\n\n{chunk}"}
            ],
            temperature=0.7,
            max_tokens=3500,
            top_p=0.9
        )
        
        result = response.choices[0].message.content
        print(f"✅ Successfully processed chunk, generated {len(result)} characters")
        return result
    except Exception as e:
        print(f"\n❌ Error processing chunk: {str(e)}")
        if "context length" in str(e).lower():
            print(f"   Chunk length (chars): {len(chunk)}")
        return None

def extract_qa_pairs(text):
    """Extract question-answer pairs from text."""
    # Match Q: and A: patterns, handling various formats
    pattern = re.compile(r'(?:^|\n)Q(?:uestion)?:\s*(.*?)(?:\n)A(?:nswer)?:\s*(.*?)(?=(?:\n\s*Q(?:uestion)?:|\n\s*$|$))', re.DOTALL)
    matches = pattern.findall(text)
    
    qa_pairs = []
    for question, answer in matches:
        # Clean up whitespace
        question = question.strip()
        answer = answer.strip()
        
        # Skip empty pairs
        if not question or not answer:
            continue
            
        qa_pairs.append({
            'question': question,
            'answer': answer
        })
    
    return qa_pairs

def process_file(input_file, output_file, max_tokens=2000, auto_chunk=True, chunk_size=None, overlap=0.2):
    """
    Process a text file to create training examples.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to save the output training data
        max_tokens (int, optional): Maximum tokens per chunk when using auto-chunking
        auto_chunk (bool, optional): Whether to use automatic chunking based on token count
        chunk_size (int, optional): Custom chunk size in characters (used when auto_chunk is False)
        overlap (float, optional): Fraction of overlap between chunks (0.0 to 1.0)
    """
    print(f"Starting to process {input_file}...")
    
    # Test API connection
    if not test_api_connection():
        print("Error: Could not connect to the API. Please check your internet connection.")
        return
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Chunk the text
    print("Chunking text...")
    if auto_chunk:
        print(f"Using automatic chunking with max_tokens={max_tokens}")
    else:
        print(f"Using custom chunk size: {chunk_size} characters with {overlap*100}% overlap")
    
    chunks = chunk_text(text, max_tokens, auto_chunk, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks.")
    
    # Process each chunk
    print("Processing chunks...")
    all_qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        result = process_chunk(chunk)
        if result:
            qa_pairs = extract_qa_pairs(result)
            print(f"Extracted {len(qa_pairs)} Q&A pairs from chunk")
            all_qa_pairs.extend(qa_pairs)
    
    # Write the results to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in all_qa_pairs:
                f.write(f"Q: {pair['question']}\nA: {pair['answer']}\n\n")
        print(f"Successfully wrote {len(all_qa_pairs)} Q&A pairs to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
    
    print("Processing complete!")

def main():
    """Command line interface for the script."""
    parser = argparse.ArgumentParser(description='Process Federalist Papers to create training examples')
    parser.add_argument('--input', '-i', default=os.environ.get('INPUT_FILE', 'federalist_papers.txt'),
                        help='Input file containing Federalist Papers text')
    parser.add_argument('--output', '-o', default=os.environ.get('OUTPUT_FILE', 'trainingdata2.txt'),
                        help='Output file for Q&A training data')
    parser.add_argument('--max-tokens', '-m', type=int, default=2000,
                        help='Maximum tokens per chunk when using auto-chunking')
    parser.add_argument('--chunk-size', '-c', type=int, 
                        help='Custom chunk size in characters (disables auto-chunking)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Fraction of overlap between chunks (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Determine if we're using auto-chunking or custom chunk size
    auto_chunk = args.chunk_size is None
    
    process_file(
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        auto_chunk=auto_chunk,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main() 