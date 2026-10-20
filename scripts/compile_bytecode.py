"""
Compile Python modules to bytecode for faster startup and reduced memory footprint.
Creates optimized .pyc files with -OO optimization level.
"""

import py_compile
import compileall
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compile_directory(directory: Path, optimize: int = 2) -> bool:
    """
    Compile all Python files in a directory to bytecode.
    
    Args:
        directory: Directory to compile
        optimize: Optimization level (0=none, 1=-O, 2=-OO)
        
    Returns:
        True if successful
    """
    logger.info(f"Compiling directory: {directory}")
    logger.info(f"Optimization level: {optimize}")
    
    try:
        # Compile with specified optimization
        success = compileall.compile_dir(
            directory,
            maxlevels=10,
            quiet=0,
            optimize=optimize,
            force=True
        )
        
        if success:
            logger.info(f"Successfully compiled {directory}")
            
            # Count compiled files
            pyc_count = sum(1 for _ in directory.rglob('*.pyc'))
            logger.info(f"Generated {pyc_count} bytecode files")
            
        return success
        
    except Exception as e:
        logger.error(f"Error compiling {directory}: {e}")
        return False


def remove_source_files(directory: Path, keep_init: bool = True):
    """
    Remove .py source files after compilation.
    
    Args:
        directory: Directory to process
        keep_init: Keep __init__.py files
    """
    logger.info("Removing source files...")
    
    removed_count = 0
    for py_file in directory.rglob('*.py'):
        # Keep __init__.py if requested
        if keep_init and py_file.name == '__init__.py':
            continue
            
        try:
            py_file.unlink()
            removed_count += 1
        except Exception as e:
            logger.warning(f"Could not remove {py_file}: {e}")
            
    logger.info(f"Removed {removed_count} source files")


def rename_bytecode_files(directory: Path):
    """
    Rename .pyc files to .py for direct execution.
    Moves files from __pycache__ to parent directory.
    """
    logger.info("Renaming bytecode files...")
    
    renamed_count = 0
    for pycache_dir in directory.rglob('__pycache__'):
        for pyc_file in pycache_dir.glob('*.pyc'):
            # Extract original name (remove cpython-311.opt-2 suffix)
            original_name = pyc_file.stem.split('.')[0]
            new_path = pycache_dir.parent / f"{original_name}.py"
            
            try:
                pyc_file.rename(new_path)
                renamed_count += 1
            except Exception as e:
                logger.warning(f"Could not rename {pyc_file}: {e}")
                
        # Remove empty __pycache__ directories
        try:
            if not list(pycache_dir.iterdir()):
                pycache_dir.rmdir()
        except:
            pass
            
    logger.info(f"Renamed {renamed_count} bytecode files")


def main():
    """Main compilation process."""
    project_root = Path(__file__).parent.parent
    
    directories_to_compile = [
        project_root / 'src',
        project_root / 'scripts',
    ]
    
    logger.info("=== Starting Bytecode Compilation ===")
    logger.info(f"Project root: {project_root}")
    
    for directory in directories_to_compile:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue
            
        # Compile to optimized bytecode (-OO removes docstrings and asserts)
        success = compile_directory(directory, optimize=2)
        
        if not success:
            logger.error(f"Failed to compile {directory}")
            sys.exit(1)
            
    logger.info("\n=== Compilation Complete ===")
    logger.info("Bytecode files generated successfully")
    
    # Optional: Remove source files in production
    if '--remove-source' in sys.argv:
        logger.info("\n=== Removing Source Files ===")
        for directory in directories_to_compile:
            if directory.exists():
                remove_source_files(directory, keep_init=True)
                
    # Optional: Rename .pyc to .py for execution
    if '--rename-pyc' in sys.argv:
        logger.info("\n=== Renaming Bytecode Files ===")
        for directory in directories_to_compile:
            if directory.exists():
                rename_bytecode_files(directory)


if __name__ == '__main__':
    main()
