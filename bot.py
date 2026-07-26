import requests
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI, Form
from twilio.rest import Client
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize vectorstore once
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./football_db",
    embedding_function=embeddings
)

load_dotenv()

BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

@tool
def get_team_info(team_name: str) -> str:
    """Get basic info about a football team."""
    response = requests.get(f"{BASE_URL}/searchteams.php?t={team_name}")
    data = response.json()
    if not data["teams"]:
        return f"Team '{team_name}' not found."
    team = data["teams"][0]
    return f"Team: {team['strTeam']}\nLeague: {team['strLeague']}\nCountry: {team['strCountry']}\nStadium: {team['strStadium']}"

@tool
def get_recent_results(team_name: str) -> str:
    """Get recent match results for a football team."""
    response = requests.get(f"{BASE_URL}/searchteams.php?t={team_name}")
    data = response.json()
    if not data["teams"]:
        return f"Team '{team_name}' not found."
    team_id = data["teams"][0]["idTeam"]
    results = requests.get(f"{BASE_URL}/eventslast.php?id={team_id}")
    events = results.json().get("results", [])
    if not events:
        return "No recent results found."
    output = "Recent results:\n"
    for e in events[-5:]:
        output += f"{e['strEvent']}: {e['intHomeScore']}-{e['intAwayScore']}\n"
    return output

@tool
def get_upcoming_fixtures(team_name: str) -> str:
    """Get upcoming fixtures for a football team."""
    response = requests.get(f"{BASE_URL}/searchteams.php?t={team_name}")
    data = response.json()
    if not data["teams"]:
        return f"Team '{team_name}' not found."
    team_id = data["teams"][0]["idTeam"]
    fixtures = requests.get(f"{BASE_URL}/eventsnext.php?id={team_id}")
    events = fixtures.json().get("events", [])
    if not events:
        return "No upcoming fixtures found."
    output = "Upcoming fixtures:\n"
    for e in events[:5]:
        output += f"{e['strEvent']} - {e['dateEvent']}\n"
    return output

@tool
def get_team_players(team_name: str) -> str:
    """Get list of players for a football team."""
    response = requests.get(f"{BASE_URL}/searchteams.php?t={team_name}")
    data = response.json()
    if not data["teams"]:
        return f"Team '{team_name}' not found."
    team_id = data["teams"][0]["idTeam"]
    players = requests.get(f"{BASE_URL}/lookup_all_players.php?id={team_id}")
    data = players.json()
    if not data.get("player"):
        return "No players found."
    output = "Players:\n"
    for p in data["player"][:15]:
        output += f"- {p['strPlayer']} ({p['strPosition']})\n"
    return output

@tool
def search_player(player_name: str) -> str:
    """Search for a football player and get their info."""
    headers = {
        "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com",
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY")
    }
    response = requests.get(
        "https://free-api-live-football-data.p.rapidapi.com/football-players-search",
        headers=headers,
        params={"search": player_name}
    )
    data = response.json()
    players = data.get("response", [])
    if not players:
        return f"No player found for '{player_name}'."
    output = "Players found:\n"
    count = 0
    for p in players:
        if count >= 5:
            break
        if isinstance(p, dict):
            name = p.get("name", "Unknown")
            team = p.get("teamName", "Unknown team")
            output += f"- {name} ({team})\n"
        else:
            output += f"- {p}\n"
        count += 1
    return output
@tool
def get_league_standings(league_name: str) -> str:
    """Get current standings/table for a football league. Supports Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League."""
    league_codes = {
        "premier league": "PL",
        "epl": "PL",
        "la liga": "PD",
        "bundesliga": "BL1",
        "serie a": "SA",
        "ligue 1": "FL1",
        "champions league": "CL",
    }
    
    code = league_codes.get(league_name.lower())
    if not code:
        return f"League '{league_name}' not supported. Try: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League."
    
    headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}
    response = requests.get(
        f"https://api.football-data.org/v4/competitions/{code}/standings",
        headers=headers
    )
    data = response.json()
    standings = data.get("standings", [])[0].get("table", [])
    if not standings:
        return "Standings not available right now."
    
    output = f"{league_name.title()} Standings:\n"
    for team in standings[:10]:
        output += f"{team['position']}. {team['team']['name']} - {team['points']} pts\n"
    return output

@tool
def get_live_scores() -> str:
    """Get current live football scores happening right now."""
    headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}
    response = requests.get(
        "https://api.football-data.org/v4/matches?status=LIVE",
        headers=headers
    )
    data = response.json()
    matches = data.get("matches", [])
    if not matches:
        return "No live matches right now."
    
    output = "Live Scores:\n"
    for m in matches[:10]:
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        home_score = m["score"]["fullTime"]["home"]
        away_score = m["score"]["fullTime"]["away"]
        output += f"{home} {home_score} - {away_score} {away}\n"
    return output

from tavily import TavilyClient

@tool
def get_transfer_news(query: str) -> str:
    """Search for latest football transfer news, injuries, and updates."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(
        query=f"football {query} 2026",
        max_results=3
    )
    if not results["results"]:
        return "No news found."
    
    output = f"Latest news on '{query}':\n"
    for r in results["results"]:
        output += f"- {r['title']}\n  {r['content'][:150]}...\n\n"
    return output

@tool
def get_top_scorers(league_name: str) -> str:
    """Get top scorers for a football league."""
    league_codes = {
        "premier league": "PL",
        "epl": "PL",
        "la liga": "PD",
        "bundesliga": "BL1",
        "serie a": "SA",
        "ligue 1": "FL1",
        "champions league": "CL",
    }
    
    code = league_codes.get(league_name.lower())
    if not code:
        return f"League '{league_name}' not supported. Try: Premier League, La Liga, Bundesliga, Serie A, Ligue 1."
    
    headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}
    response = requests.get(
        f"https://api.football-data.org/v4/competitions/{code}/scorers",
        headers=headers,
        params={"season": "2025"}
    )
    data = response.json()
    scorers = data.get("scorers", [])
    if not scorers:
        return "No scorers data available."
    
    output = f"Top Scorers - {league_name.title()}:\n"
    for i, s in enumerate(scorers[:10], 1):
        name = s["player"]["name"]
        team = s["team"]["shortName"]
        goals = s["goals"]
        assists = s.get("assists", 0)
        output += f"{i}. {name} ({team}) - {goals} goals, {assists} assists\n"
    return output

@tool
def get_player_injury(player_name: str) -> str:
    """Get latest injury news and status for a football player."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(
        query=f"{player_name} injury update 2026",
        max_results=3
    )
    if not results["results"]:
        return f"No injury news found for {player_name}."
    
    output = f"Injury update for {player_name}:\n"
    for r in results["results"]:
        output += f"- {r['title']}\n  {r['content'][:200]}...\n\n"
    return output

@tool
def predict_match(team1: str, team2: str) -> str:
    """Predict the outcome of a football match between two teams based on recent form and stats."""
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Search for recent form of both teams
    results = tavily.search(
        query=f"{team1} vs {team2} prediction 2026 form stats",
        max_results=3
    )
    
    if not results["results"]:
        return "Not enough data to make a prediction."
    
    output = f"Match Preview: {team1} vs {team2}\n\n"
    for r in results["results"]:
        output += f"- {r['title']}\n  {r['content'][:200]}...\n\n"
    return output

@tool
def search_football_knowledge(question: str) -> str:
    """Search football rules, regulations and knowledge base to answer questions."""
    results = vectorstore.similarity_search(question, k=2)
    if not results:
        return "No relevant information found."
    
    output = "From football knowledge base:\n"
    for r in results:
        output += f"{r.page_content}\n\n"
    return output

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

agent = create_react_agent(
    model=model,
    tools=[get_team_info, get_recent_results, get_upcoming_fixtures, get_team_players, search_player, get_league_standings, get_live_scores, get_transfer_news, get_top_scorers, get_player_injury, predict_match, search_football_knowledge],
    prompt="You are a football sports assistant on WhatsApp. STRICT RULES: 1) For ANY question about football rules, offside, VAR, cards, penalties - ALWAYS call search_football_knowledge tool FIRST before answering. 2) For team info - use get_team_info. 3) For standings - use get_league_standings. 4) For scores - use get_live_scores. 5) For transfers/news - use get_transfer_news. 6) For predictions - use predict_match. 7) NEVER answer from your own memory. ALWAYS use a tool first.",
    checkpointer=memory,
)

def ask_agent(message: str, user_id: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    return result["messages"][-1].content

app = FastAPI()

@app.post("/webhook")
async def webhook(Body: str = Form(), From: str = Form()):
    try:
        response = ask_agent(Body, From)
        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        client.messages.create(
            from_="whatsapp:+14155238886",
            to=From,
            body=response
        )
    except Exception as e:
        print(f"Error: {e}")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)