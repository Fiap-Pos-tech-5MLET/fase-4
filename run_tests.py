#!/usr/bin/env python3
"""
Script para executar todos os testes com cobertura.

Uso:
    python run_tests.py              # Executa tudo
    python run_tests.py --coverage   # Com relatório de cobertura
    python run_tests.py --html       # Gera relatório HTML
    python run_tests.py --fast       # Testes em paralelo (rápido)
    python run_tests.py --check      # Apenas verificar cobertura
"""

import subprocess
import sys
import os
from pathlib import Path

# Cores para output
class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Imprime header formatado."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Imprime mensagem de sucesso."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Imprime mensagem de erro."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Imprime mensagem de info."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def run_command(cmd: list, description: str) -> bool:
    """Executa comando e retorna sucesso/falha."""
    print_info(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print_success(f"{description}")
        return True
    else:
        print_error(f"{description} - FALHOU!")
        return False


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Executar testes com cobertura",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_tests.py              # Executa tudo
  python run_tests.py --coverage   # Com cobertura
  python run_tests.py --html       # Relatório HTML
  python run_tests.py --fast       # Em paralelo (rápido)
  python run_tests.py --check      # Verifica cobertura
        """
    )
    
    parser.add_argument('--coverage', action='store_true', 
                       help='Mostrar relatório de cobertura')
    parser.add_argument('--html', action='store_true',
                       help='Gerar relatório HTML (htmlcov/index.html)')
    parser.add_argument('--fast', action='store_true',
                       help='Executar testes em paralelo')
    parser.add_argument('--check', action='store_true',
                       help='Apenas verificar cobertura')
    
    args = parser.parse_args()
    
    print_header("Stock Prediction API - Test Suite")
    
    # Build pytest command
    pytest_cmd = ['pytest', 'tests/', '-v', '--tb=short']
    
    if args.coverage or args.html or args.check:
        pytest_cmd.extend([
            '--cov=src',
            '--cov=app',
            '--cov-report=term-missing',
            '--cov-report=xml'
        ])
    
    if args.html:
        pytest_cmd.append('--cov-report=html')
    
    if args.fast:
        pytest_cmd.extend(['-n', 'auto'])
        print_info("Executando testes em paralelo com pytest-xdist\n")
    
    # Run tests
    success = run_command(pytest_cmd, "Testes executados")
    
    if not success:
        print_error("Testes falharam!")
        sys.exit(1)
    
    # Check coverage if requested
    if args.check:
        print_header("Verificando Cobertura (mínimo 90%)")
        check_cmd = ['coverage', 'report', '--fail-under=90']
        if not run_command(check_cmd, "Cobertura está acima de 90%"):
            sys.exit(1)
    
    # Show coverage report if requested
    if args.coverage:
        print_header("Relatório de Cobertura")
        coverage_cmd = ['coverage', 'report']
        run_command(coverage_cmd, "Relatório exibido")
    
    # Generate HTML if requested
    if args.html:
        print_header("Gerando Relatório HTML")
        print_info("Abrindo: htmlcov/index.html")
        html_cmd = ['coverage', 'html']
        run_command(html_cmd, "Relatório HTML gerado")
        
        # Try to open browser
        if sys.platform == 'darwin':
            os.system('open htmlcov/index.html')
        elif sys.platform == 'linux':
            os.system('xdg-open htmlcov/index.html')
        elif sys.platform == 'win32':
            os.system('start htmlcov\\index.html')
    
    print_header("✓ Testes Concluídos com Sucesso!")
    
    # Print summary
    print(f"{Colors.GREEN}{Colors.BOLD}Resumo:{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓ Testes: PASSOU{Colors.ENDC}")
    if args.coverage or args.check:
        print(f"  {Colors.GREEN}✓ Cobertura: >= 90%{Colors.ENDC}")
    if args.html:
        print(f"  {Colors.GREEN}✓ Relatório HTML: htmlcov/index.html{Colors.ENDC}")
    print()


if __name__ == '__main__':
    main()
