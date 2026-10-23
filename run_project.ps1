# PowerShell Launcher for Encrypted Traffic GNN Project
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host " FINE-GRAINED ENCRYPTED TRAFFIC CLASSIFICATION (GraphSAGE / GAT)" -ForegroundColor Green
Write-Host " Windows PowerShell Launcher" -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " 1. Run Fine-Grained Application Experiment"
Write-Host " 2. Run Binary VPN vs Non-VPN Experiment"
Write-Host " 3. Launch Demonstration Interface (GUI)"
Write-Host " 4. Launch Demonstration Interface (CLI Sample Mode)"
Write-Host " 5. Run Complete Unit Test Suite"
Write-Host " 6. View Benchmark Dataset Catalog"
Write-Host " 7. Exit"
Write-Host ""

$choice = Read-Host "Select an option [1-7]"

switch ($choice) {
    "1" { python -m src.pipeline --experiment fine_grained_app }
    "2" { python -m src.pipeline --experiment binary_vpn }
    "3" { python app.py }
    "4" { python app.py --cli }
    "5" { python -m unittest discover tests -v }
    "6" { python -m src.pipeline --show_catalog }
    "7" { exit }
    Default { Write-Host "Invalid option selected." -ForegroundColor Red }
}
