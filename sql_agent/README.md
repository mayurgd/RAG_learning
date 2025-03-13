python==3.12.0

create env
conda create -n agentic_ai python==3.12
conda install --yes --file requirements.txt

change current directory to sql_agent
set db uri in crew.py script
execute the code: python -m src.sql_agent.main