"""
Multi-Agent System — entry point.

Usage:
  python main.py "Research the rise of agentic AI in 2025"
  python main.py "Analyze data/sales.csv"
  python main.py "Research AI chip supply chain and analyze data/chip_data.csv"
"""
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from agents.orchestrator import route

load_dotenv()
console = Console()


def run(user_request: str):
    console.print(Panel(f"[bold cyan]Request:[/bold cyan] {user_request}", expand=False))

    # Step 1: Route the request
    console.print("\n[bold yellow]Orchestrator[/bold yellow] → classifying task...")
    task = route(user_request)
    console.print(f"  Task type : [green]{task['task_type']}[/green]")
    console.print(f"  Report    : [green]{task['report_title']}[/green]")
    if task.get("research_topic"):
        console.print(f"  Topic     : {task['research_topic']}")
    if task.get("csv_path"):
        console.print(f"  CSV       : {task['csv_path']}")

    results = {}

    # Step 2: Run Research Agent if needed
    if task["task_type"] in ("research", "combined"):
        from agents.research_agent import ResearchAgent
        console.print("\n[bold yellow]Research Agent[/bold yellow] → searching and reading...")
        agent = ResearchAgent()
        results["research"] = agent.run(task["research_topic"])
        console.print("  [green]✓ Research complete[/green]")

    # Step 3: Run Data Agent if needed
    if task["task_type"] in ("data_analysis", "combined"):
        from agents.data_agent import DataAgent
        console.print("\n[bold yellow]Data Agent[/bold yellow] → analyzing CSV...")
        agent = DataAgent()
        results["data"] = agent.run(task["csv_path"])
        console.print("  [green]✓ Data analysis complete[/green]")

    # Step 4: Write report
    from agents.report_writer import write_report
    console.print("\n[bold yellow]Report Writer[/bold yellow] → generating report...")
    report_path = write_report(task["report_title"], results)
    console.print(f"  [green]✓ Report saved → {report_path}[/green]")

    console.print(Panel(f"[bold green]Done![/bold green] Report at: {report_path}", expand=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
