"""Rich-based debug output for Scout detections."""
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

console = Console()

def create_progress() -> Progress:
    """Create a Rich progress bar for Scout operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

def print_detection_start(image_path: str, config: Dict[str, Any]) -> None:
    """Print detection start information."""
    console.rule("[bold blue]Scout Detection Started")
    
    config_table = Table(show_header=False)
    config_table.add_row("Image Path", image_path)
    for key, value in config.items():
        if key != "image_path":
            config_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(Panel(config_table, title="Configuration"))

def print_yolo_results(num_detections: int, target_list: List[str], source: str) -> None:
    """Print YOLO detection results."""
    detection_table = Table(title="YOLO Detection Results")
    detection_table.add_column("Metric", style="cyan")
    detection_table.add_column("Value", style="green")
    
    detection_table.add_row("Objects Detected", str(num_detections))
    detection_table.add_row("Target Source", source)
    detection_table.add_row("Target Objects", ", ".join(target_list))
    
    console.print(detection_table)

def print_iteration_results(
    iteration: int,
    max_iterations: int,
    detections: int,
    removed: int,
    confidence: float,
    improvements: List[str],
    issues: List[str]
) -> None:
    """Print iteration results in a structured format."""
    console.rule(f"[bold blue]Iteration {iteration}/{max_iterations}")
    
    # Stats table
    stats = Table(title="Statistics", show_header=False)
    stats.add_row("Detections", str(detections))
    stats.add_row("Removed", str(removed))
    stats.add_row("Confidence", f"{confidence:.2f}")
    console.print(stats)
    
    # Improvements
    if improvements:
        console.print("[bold green]Improvements:")
        for imp in improvements:
            console.print(f"✓ {imp}")
    
    # Issues
    if issues:
        console.print("[bold yellow]Remaining Issues:")
        for issue in issues:
            console.print(f"! {issue}")

def print_final_results(results: Dict[str, Any]) -> None:
    """Print final detection results."""
    console.rule("[bold blue]Detection Complete")
    
    # Final stats table
    stats = Table(title="Final Results", show_header=False)
    stats.add_row("Processing Time", f"{results['processing_time']:.2f}s")
    stats.add_row("Final Detections", str(len(results['detections'])))
    stats.add_row("Removed Detections", str(len(results['removed_detections'])))
    stats.add_row("Target Source", results['target_source'])
    
    if results.get('scene_description'):
        stats.add_row("Scene Description", results['scene_description'])
    
    console.print(stats)
    
    # Print iteration history summary
    history_table = Table(title="Iteration History")
    history_table.add_column("Iteration", style="cyan")
    history_table.add_column("Detections", style="green")
    history_table.add_column("Removed", style="yellow")
    history_table.add_column("Confidence", style="blue")
    
    for h in results['iteration_history']:
        history_table.add_row(
            str(h['iteration']),
            str(h['detections_count']),
            str(h['removed_count']),
            f"{h['confidence_score']:.2f}"
        )
    
    console.print(history_table)
