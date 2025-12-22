"""Progress bar utilities for LatentMAS project.

This module provides progress bar functionality that works seamlessly with
the enhanced logging system, keeping the progress bar at the bottom while
logs scroll above it.
"""

import sys
from typing import Optional, Any
from tqdm import tqdm


class ProgressBarManager:
    """Manager for progress bars that integrates with logging.
    
    This class provides a clean interface for creating and managing progress bars
    that stay at the bottom of the console while logs appear above them.
    """
    
    def __init__(self):
        """Initialize the progress bar manager."""
        self.active_bars = {}
        self.main_bar = None
    
    def create_main_progress(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "sample"
    ) -> tqdm:
        """Create the main progress bar for overall processing.
        
        Args:
            total: Total number of items to process
            desc: Description for the progress bar
            unit: Unit name for items being processed
            
        Returns:
            tqdm progress bar instance
        """
        if self.main_bar is not None:
            self.main_bar.close()
        
        # Create progress bar with specific formatting
        self.main_bar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            position=0,
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stderr,
            dynamic_ncols=True
        )
        
        return self.main_bar
    
    def update_main_progress(self, n: int = 1, **kwargs) -> None:
        """Update the main progress bar.
        
        Args:
            n: Number of items to increment by
            **kwargs: Additional parameters to update (e.g., postfix)
        """
        if self.main_bar is not None:
            self.main_bar.update(n)
            if kwargs:
                self.main_bar.set_postfix(**kwargs)
    
    def set_main_description(self, desc: str) -> None:
        """Update the main progress bar description.
        
        Args:
            desc: New description text
        """
        if self.main_bar is not None:
            self.main_bar.set_description(desc)
    
    def set_main_postfix(self, **kwargs) -> None:
        """Set postfix information on the main progress bar.
        
        Args:
            **kwargs: Key-value pairs to display (e.g., acc=0.85, loss=0.23)
        """
        if self.main_bar is not None:
            self.main_bar.set_postfix(**kwargs)
    
    def create_sub_progress(
        self,
        name: str,
        total: int,
        desc: str = "Processing",
        position: int = 1
    ) -> tqdm:
        """Create a sub-progress bar for nested operations.
        
        Args:
            name: Unique name for this progress bar
            total: Total number of items
            desc: Description text
            position: Position index (higher numbers appear lower)
            
        Returns:
            tqdm progress bar instance
        """
        if name in self.active_bars:
            self.active_bars[name].close()
        
        bar = tqdm(
            total=total,
            desc=desc,
            position=position,
            leave=False,
            ncols=100,
            file=sys.stderr,
            dynamic_ncols=True
        )
        
        self.active_bars[name] = bar
        return bar
    
    def update_sub_progress(self, name: str, n: int = 1) -> None:
        """Update a sub-progress bar.
        
        Args:
            name: Name of the progress bar to update
            n: Number of items to increment by
        """
        if name in self.active_bars:
            self.active_bars[name].update(n)
    
    def close_sub_progress(self, name: str) -> None:
        """Close a sub-progress bar.
        
        Args:
            name: Name of the progress bar to close
        """
        if name in self.active_bars:
            self.active_bars[name].close()
            del self.active_bars[name]
    
    def close_all(self) -> None:
        """Close all progress bars."""
        # Close sub bars first
        for bar in self.active_bars.values():
            bar.close()
        self.active_bars.clear()
        
        # Close main bar
        if self.main_bar is not None:
            self.main_bar.close()
            self.main_bar = None
    
    def write(self, message: str) -> None:
        """Write a message above the progress bar.
        
        Args:
            message: Message to write
        """
        if self.main_bar is not None:
            self.main_bar.write(message)
        else:
            print(message, file=sys.stderr)


# Global progress bar manager instance
_global_progress_manager: Optional[ProgressBarManager] = None


def get_progress_manager() -> ProgressBarManager:
    """Get the global progress bar manager instance.
    
    Returns:
        Global ProgressBarManager instance
    """
    global _global_progress_manager
    if _global_progress_manager is None:
        _global_progress_manager = ProgressBarManager()
    return _global_progress_manager


def reset_progress_manager() -> None:
    """Reset the global progress bar manager.
    
    This should be called at the start of a new run to ensure clean state.
    """
    global _global_progress_manager
    if _global_progress_manager is not None:
        _global_progress_manager.close_all()
    _global_progress_manager = ProgressBarManager()


class ProgressContext:
    """Context manager for progress bars.
    
    This ensures progress bars are properly cleaned up even if errors occur.
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "sample",
        manager: Optional[ProgressBarManager] = None
    ):
        """Initialize the progress context.
        
        Args:
            total: Total number of items
            desc: Description text
            unit: Unit name
            manager: Optional progress manager (uses global if None)
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.manager = manager or get_progress_manager()
        self.bar = None
    
    def __enter__(self) -> tqdm:
        """Enter the context and create the progress bar.
        
        Returns:
            Progress bar instance
        """
        self.bar = self.manager.create_main_progress(
            total=self.total,
            desc=self.desc,
            unit=self.unit
        )
        return self.bar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up the progress bar.
        
        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.bar is not None:
            self.bar.close()
        return False  # Don't suppress exceptions

