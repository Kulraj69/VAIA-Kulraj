#!/usr/bin/env python3
"""
Script to decode AWS Bedrock API key and list available models.
"""
import base64
import json
import os
import sys
from typing import Optional, Dict, Any

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("Installing boto3...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError


# BEDROCK_KEY should be provided as environment variable or command line argument
# Example: export BEDROCK_KEY="your-key-here" or pass as command line argument
BEDROCK_KEY: str = os.environ.get("BEDROCK_KEY", "")


def decode_bedrock_key(encoded_key: str) -> Optional[Dict[str, str]]:
    """Decode the base64 encoded AWS Bedrock key."""
    try:
        decoded_bytes = base64.b64decode(encoded_key)
        print(f"Decoded bytes length: {len(decoded_bytes)}")
        print(f"First 20 bytes (hex): {decoded_bytes[:20].hex()}")
        
        # Try UTF-8 decode
        try:
            decoded_str = decoded_bytes.decode('utf-8')
            print(f"Decoded as UTF-8: {decoded_str}")
            
            # Try to parse as JSON if it's JSON format
            try:
                return json.loads(decoded_str)
            except json.JSONDecodeError:
                # If not JSON, might be key:secret format
                if ':' in decoded_str:
                    parts = decoded_str.split(':', 1)
                    return {
                        "access_key_id": parts[0],
                        "secret_access_key": parts[1] if len(parts) > 1 else ""
                    }
                return {"raw_key": decoded_str}
        except UnicodeDecodeError:
            # Binary data - try to extract credentials using common patterns
            print("Decoded data is binary, attempting to extract credentials...")
            decoded_hex = decoded_bytes.hex()
            print(f"Hex representation: {decoded_hex[:100]}...")
            
            # Try to extract readable ASCII strings from binary data
            readable_strings = []
            current_string = ""
            for byte in decoded_bytes:
                if 32 <= byte <= 126:  # Printable ASCII range
                    current_string += chr(byte)
                else:
                    if len(current_string) >= 10:  # Only keep strings of length 10+
                        readable_strings.append(current_string)
                    current_string = ""
            if current_string:
                readable_strings.append(current_string)
            
            print(f"Extracted readable strings: {readable_strings}")
            
            # Look for patterns that might be access key ID and secret
            # The format might be: "BedrockAPIKey-XXXX-at-YYYY:secret"
            for s in readable_strings:
                if ':' in s and len(s) > 20:
                    parts = s.split(':', 1)
                    if len(parts) == 2:
                        print(f"Found potential credentials format with ':' separator")
                        # The first part might be access key ID pattern
                        access_key_part = parts[0]
                        secret_part = parts[1]
                        # Try to extract actual access key ID (might start with AKIA or similar)
                        if '-' in access_key_part:
                            # Format might be "BedrockAPIKey-XXXX-at-YYYY"
                            # Extract the actual key ID
                            key_parts = access_key_part.split('-')
                            potential_key_id = None
                            for part in key_parts:
                                if len(part) >= 16 and not part.startswith('Bedrock'):
                                    potential_key_id = part
                                    break
                            
                            # Try decoding the secret (might be base64 encoded)
                            secret_to_use = secret_part
                            try:
                                decoded_secret = base64.b64decode(secret_part)
                                secret_str = decoded_secret.decode('utf-8')
                                print(f"Secret decoded from base64: {secret_str}")
                                secret_to_use = secret_str
                            except:
                                print("Secret not base64 encoded, using as-is")
                            
                            # Extract potential AWS access key ID from the format
                            # Format: BedrockAPIKey-814k-at-701544683046
                            # Try different parts as the actual access key ID
                            aws_key_id = None
                            if '814k' in key_parts:
                                aws_key_id = '814k'
                            elif '701544683046' in key_parts:
                                aws_key_id = '701544683046'
                            
                            return {
                                "access_key_id": aws_key_id or potential_key_id or access_key_part,
                                "secret_access_key": secret_to_use,
                                "original_format": access_key_part,
                                "alternative_key_ids": key_parts,
                                "all_alternatives": [aws_key_id, potential_key_id, access_key_part] + key_parts if aws_key_id else [potential_key_id, access_key_part] + key_parts
                            }
                        else:
                            # The secret might be base64 encoded, try decoding it
                            try:
                                decoded_secret = base64.b64decode(secret_part)
                                secret_str = decoded_secret.decode('utf-8')
                                print(f"Secret decoded from base64: {secret_str}")
                                return {
                                    "access_key_id": access_key_part,
                                    "secret_access_key": secret_str
                                }
                            except:
                                pass
                            
                            return {
                                "access_key_id": access_key_part,
                                "secret_access_key": secret_part,
                                "potential_base64_secret": True
                            }
            
            # If no clear pattern found, try decoding as different encodings
            for encoding in ['latin1', 'iso-8859-1']:
                try:
                    decoded_str = decoded_bytes.decode(encoding)
                    if ':' in decoded_str:
                        parts = decoded_str.split(':', 1)
                        return {
                            "access_key_id": parts[0],
                            "secret_access_key": parts[1] if len(parts) > 1 else ""
                        }
                except:
                    pass
            
            return {
                "raw_bytes": decoded_bytes.hex(),
                "readable_strings": readable_strings,
                "note": "Binary data - extracted readable strings above"
            }
    except Exception as e:
        print(f"Error decoding key: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_bedrock_models(
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    region: str = "us-east-1"
) -> list:
    """List available models in AWS Bedrock."""
    try:
        # Create Bedrock client
        if access_key_id and secret_access_key:
            session = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region
            )
        else:
            # Try using default credentials
            session = boto3.Session(region_name=region)
        
        bedrock_client = session.client('bedrock')
        
        # List foundation models
        print(f"\nListing Bedrock models in region: {region}")
        print("=" * 60)
        
        response = bedrock_client.list_foundation_models()
        models = response.get('modelSummaries', [])
        
        if not models:
            print("No models found. Trying different regions...")
            # Try common regions
            regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            for reg in regions:
                if reg == region:
                    continue
                try:
                    session = boto3.Session(
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key,
                        region_name=reg
                    ) if access_key_id and secret_access_key else boto3.Session(region_name=reg)
                    bedrock_client = session.client('bedrock')
                    response = bedrock_client.list_foundation_models()
                    models = response.get('modelSummaries', [])
                    if models:
                        print(f"Found models in region: {reg}")
                        break
                except Exception as e:
                    print(f"Error checking region {reg}: {e}")
                    continue
        
        return models
        
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please provide access_key_id and secret_access_key.")
        return []
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        print(f"AWS Client Error ({error_code}): {error_msg}")
        return []
    except Exception as e:
        print(f"Error accessing Bedrock: {e}")
        return []


def format_model_info(models: list) -> None:
    """Format and display model information."""
    if not models:
        print("\nNo models found.")
        return
    
    print(f"\nFound {len(models)} model(s):\n")
    print("-" * 60)
    
    for idx, model in enumerate(models, 1):
        model_id = model.get('modelId', 'N/A')
        model_name = model.get('modelName', 'N/A')
        provider = model.get('providerName', 'N/A')
        model_arn = model.get('modelArn', 'N/A')
        inference_types = model.get('inferenceTypesSupported', [])
        
        print(f"\n{idx}. Model ID: {model_id}")
        print(f"   Name: {model_name}")
        print(f"   Provider: {provider}")
        print(f"   ARN: {model_arn}")
        print(f"   Inference Types: {', '.join(inference_types) if inference_types else 'N/A'}")
        
        # Additional metadata
        if 'inputModalities' in model:
            print(f"   Input Modalities: {', '.join(model['inputModalities'])}")
        if 'outputModalities' in model:
            print(f"   Output Modalities: {', '.join(model['outputModalities'])}")
    
    print("\n" + "-" * 60)


def main():
    """Main function to decode key and list models."""
    print("AWS Bedrock Model Discovery")
    print("=" * 60)
    
    # Check if key is provided
    if not BEDROCK_KEY:
        print("\nError: BEDROCK_KEY not provided.")
        print("Please set it as an environment variable:")
        print('  export BEDROCK_KEY="your-key-here"')
        print("Or pass it as a command line argument:")
        print("  python3 check_bedrock_models.py <your-key>")
        if len(sys.argv) > 1:
            bedrock_key = sys.argv[1]
        else:
            return
    
    # Use command line argument if provided, otherwise use env var
    bedrock_key = sys.argv[1] if len(sys.argv) > 1 else BEDROCK_KEY
    
    # Decode the key
    print("\nStep 1: Decoding Bedrock key...")
    decoded = decode_bedrock_key(bedrock_key)
    
    if not decoded:
        print("Failed to decode key.")
        return
    
    print("\nStep 2: Extracting credentials...")
    access_key_id = decoded.get("access_key_id") or decoded.get("aws_access_key_id")
    secret_access_key = decoded.get("secret_access_key") or decoded.get("aws_secret_access_key")
    
    # If we have alternative key IDs, try them too
    alternative_key_ids = decoded.get("alternative_key_ids", [])
    
    print(f"Extracted Access Key ID: {access_key_id}")
    print(f"Extracted Secret Access Key: {'*' * min(len(str(secret_access_key)), 20)}...")
    
    if not access_key_id or not secret_access_key:
        print("Warning: Could not extract access key ID and secret access key from decoded string.")
        print("\nThe decoded key appears to be in a different format.")
        print("Attempting alternative methods...")
        
        # Try using the decoded string directly as an API key if it exists
        raw_key = decoded.get("raw_key")
        if raw_key:
            print(f"\nUsing raw key directly (may be an API key format):")
            print(f"Key length: {len(raw_key)}")
            # If the raw key looks like it might be credentials, try splitting it
            if ':' in raw_key or '-' in raw_key:
                # Might be a composite key, try to use it directly with boto3
                print("Attempting to use raw key as credentials...")
        else:
            print("Attempting to use default AWS credentials or environment variables...")
        access_key_id = None
        secret_access_key = None
    
    # Try to list models in common regions
    print("\nStep 3: Listing available models...")
    regions_to_try = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"]
    models_found = []
    
    # Try with extracted credentials first
    if access_key_id and secret_access_key:
        print(f"\nTrying with extracted credentials...")
        for region in regions_to_try:
            print(f"\nTrying region: {region}...")
            models = list_bedrock_models(access_key_id, secret_access_key, region)
            if models:
                models_found = models
                print(f"Successfully found models in {region}!")
                break
        
        # If that didn't work, try with alternative key IDs
        if not models_found and alternative_key_ids:
            print(f"\nTrying with alternative access key IDs...")
            for alt_key_id in alternative_key_ids:
                if alt_key_id and len(str(alt_key_id)) >= 8:
                    print(f"Trying alternative key ID: {alt_key_id}...")
                    for region in regions_to_try[:2]:  # Just try first 2 regions
                        models = list_bedrock_models(alt_key_id, secret_access_key, region)
                        if models:
                            models_found = models
                            print(f"Successfully found models in {region} with key ID {alt_key_id}!")
                            break
                    if models_found:
                        break
    
    # If still no models, try all alternative combinations
    if not models_found and decoded.get("all_alternatives"):
        print(f"\nTrying all alternative key ID formats...")
        all_alts = decoded.get("all_alternatives", [])
        for alt_key in all_alts:
            if alt_key and isinstance(alt_key, str) and len(alt_key) >= 8:
                print(f"Trying: {alt_key[:20]}...")
                models = list_bedrock_models(alt_key, secret_access_key, "us-east-1")
                if models:
                    models_found = models
                    break
    
    if models_found:
        format_model_info(models_found)
    else:
        print("\nCould not find models. Possible issues:")
        print("- Invalid credentials")
        print("- Bedrock not enabled in the account")
        print("- Region not supported")
        print("- Permissions issue")


if __name__ == "__main__":
    main()

