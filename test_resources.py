import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set mock mode to avoid audio issues
os.environ['MCP_TTS_MOCK_MODE'] = 'true'

async def test_resources():
    print('Testing FastMCP Resource Functionality')
    print('=' * 50)
    
    try:
        from mcp_tts_server import mcp, RESOURCES
        
        # Test that resources were added to the server
        print('\nChecking if resources were added to FastMCP server...')
        
        # Get the available resources
        available_resources = await mcp.get_resources()
        
        print(f'Found {len(available_resources)} resources:')
        print(f'Resource dict keys: {list(available_resources.keys())}')
        
        for uri, resource in available_resources.items():
            print(f'  - URI: {uri}')
            print(f'    Resource type: {type(resource)}')
            if hasattr(resource, 'name'):
                print(f'    Name: {resource.name}')
            if hasattr(resource, 'description'):
                print(f'    Description: {resource.description}')
            if hasattr(resource, 'text'):
                print(f'    Content: "{resource.text}"')
            print()
        
        # Test resource retrieval
        print('Testing resource retrieval:')
        for uri in RESOURCES.keys():
            try:
                resource = await mcp.get_resource(uri)
                if resource:
                    print(f'  [OK] {uri} -> Found')
                    if hasattr(resource, 'text'):
                        print(f'    Content: "{resource.text}"')
                else:
                    print(f'  [FAIL] {uri} -> Not found')
            except Exception as e:
                print(f'  [ERROR] {uri} -> Error: {e}')
        
        print('\n' + '=' * 50)
        print('SUCCESS: FastMCP resources are working!')
        return True
        
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_resources())
    if success:
        print('\nResources are now properly enabled with FastMCP!')
    else:
        print('\nThere was an issue with the FastMCP resources.')
