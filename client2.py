import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re 

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        
        # Use Gemini to generate a prompt for the tools

        prompt = f"""
        {query}

        Available tools:
        {json.dumps(available_tools, indent=2)}

        If the query requires a tool, include '[Use tool: <tool_name>]' in your response, followed by your explanation.
        Otherwise, provide a direct answer without tool references.
        """
        response = self.gemini.generate_content(prompt)
        final_text = [response.text]

        # Process response and handle tool calls
        #final_text = []

        if "employee" in query.lower() or "create" in query.lower() or "[Use tool: create_employee]" in response.text:
            tool_name = "create_employee"
            tool_args = {}

            # Parse employee_name (after "name" or "named", up to comma or next keyword)
            name_match = re.search(r'(?:name|named)\s+([^,]+?)(?:,|\s+nif|\s+date|\s+and|$)', query, re.IGNORECASE)
            tool_args["employee_name"] = name_match.group(1).strip() if name_match else "John Doe"
            print(f"Parsed name: {tool_args['employee_name']}")

            # Parse nif (after "nif", any digits)
            nif_match = re.search(r'nif\s+(\d+)', query, re.IGNORECASE)
            tool_args["nif"] = int(nif_match.group(1)) if nif_match else 0
            print(f"Parsed nif: {tool_args['nif']}")

            # Parse date_of_birth (DD/MM/YYYY after "date of birth", convert to YYYY-MM-DD)
            date_match = re.search(r'date\s+of\s+birth\s+(\d{2}/\d{2}/\d{4})', query, re.IGNORECASE)
            if date_match:
                day, month, year = date_match.group(1).split('/')
                tool_args["date_of_birth"] = f"{year}-{month}-{day}"
            else:
                tool_args["date_of_birth"] = ""
            print(f"Parsed date_of_birth: {tool_args['date_of_birth']}")

            # Parse address (after "address" or "and address", to end or next keyword)
            addr_match = re.search(r'(?:and\s+)?address\s+(.+)$', query, re.IGNORECASE)
            tool_args["address"] = addr_match.group(1).strip() if addr_match else ""
            print(f"Parsed address: {tool_args['address']}")

            # Ensure required field
            if not tool_args["employee_name"] or tool_args["employee_name"] == "John Doe":
                final_text.append("Warning: Employee name not parsed correctly. Using default 'John Doe'.")

            result = await self.session.call_tool(tool_name, tool_args)
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]\nResult: {result.content if hasattr(result, 'content') else result}")
            
            follow_up_prompt = f"{query}\nTool result: {result.content if hasattr(result, 'content') else result}"
            follow_up_response = self.gemini.generate_content(follow_up_prompt)
            final_text.append(follow_up_response.text)
        
        elif "[Use tool: sum_numbers]" in response.text or "add" in query.lower() or "sum" in query.lower() or "plus" in query.lower():
            tool_name = "sum_numbers"
            # Extract numbers from the query
            numbers = [int(num) for num in re.findall(r'\d+', query)]
            if len(numbers) >= 2:
                tool_args = {"number1": numbers[0], "number2": numbers[1]}
            else:
                # Fallback: Use defaults or ask Gemini for clarification
                tool_args = {"number1": 5, "number2": 3}
                final_text.append("Couldn't find two numbers in the query; using defaults 5 and 3.")
            
            result = await self.session.call_tool(tool_name, tool_args)
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]\nResult: {result}")
            
            # Follow-up with Gemini
            follow_up_prompt = f"{query}\nTool result: {result}"
            follow_up_response = self.gemini.generate_content(follow_up_prompt)
            final_text.append(follow_up_response.text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())