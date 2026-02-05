"""
Input validation utilities for QDT Nexus API.

Author: Artemis (with Claude Code)
Date: 2026-02-03

Security: Prevents injection attacks, path traversal, and DoS via malformed input.
"""

import re
from datetime import datetime
from typing import Optional, Tuple


class ValidationError(Exception):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class InputValidator:
    """
    Validates and sanitizes user inputs.
    
    All methods raise ValidationError on invalid input.
    """
    
    # Valid asset names (whitelist)
    VALID_ASSETS = {
        "Crude_Oil", "Bitcoin", "Gold", "Silver", "Natural_Gas",
        "Copper", "Platinum", "Palladium", "Wheat", "Corn",
        "Soybeans", "Coffee", "Sugar", "Cotton", "Nikkei_225"
    }
    
    # Asset ID mapping
    ASSET_IDS = {
        "1866": "Crude_Oil",
        "1860": "Bitcoin", 
        "1861": "Gold",
        "1862": "Silver",
        "1863": "Natural_Gas",
        "1864": "Copper",
        "1865": "Platinum",
        "1867": "Palladium",
        "1868": "Wheat",
        "1869": "Corn",
        "1870": "Soybeans",
        "1871": "Coffee",
        "1872": "Sugar",
        "1873": "Cotton",
        "358": "Nikkei_225"
    }
    
    @staticmethod
    def validate_asset_name(asset_name: str) -> str:
        """
        Validate asset name against whitelist.
        
        Prevents path traversal and injection attacks.
        
        Args:
            asset_name: The asset name to validate
            
        Returns:
            str: Validated asset name
            
        Raises:
            ValidationError: If asset name is invalid
        """
        if not asset_name:
            raise ValidationError("Asset name is required", "asset_name")
        
        # Strip whitespace
        asset_name = asset_name.strip()
        
        # Check against whitelist
        if asset_name not in InputValidator.VALID_ASSETS:
            # Check if it's an asset ID
            if asset_name in InputValidator.ASSET_IDS:
                return InputValidator.ASSET_IDS[asset_name]
            
            raise ValidationError(
                f"Invalid asset name: {asset_name}. "
                f"Valid assets: {', '.join(sorted(InputValidator.VALID_ASSETS))}",
                "asset_name"
            )
        
        return asset_name
    
    @staticmethod
    def validate_date(
        date_str: Optional[str], 
        param_name: str = "date",
        allow_none: bool = True
    ) -> Optional[datetime]:
        """
        Validate date string in YYYY-MM-DD format.
        
        Args:
            date_str: Date string to validate
            param_name: Parameter name for error messages
            allow_none: Whether None/empty is allowed
            
        Returns:
            datetime or None
            
        Raises:
            ValidationError: If date format is invalid
        """
        if date_str is None or date_str.strip() == "":
            if allow_none:
                return None
            raise ValidationError(f"{param_name} is required", param_name)
        
        date_str = date_str.strip()
        
        # Strict format check (prevents injection)
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            raise ValidationError(
                f"Invalid {param_name} format. Expected YYYY-MM-DD, got: {date_str}",
                param_name
            )
        
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValidationError(f"Invalid {param_name}: {str(e)}", param_name)
        
        # Reasonable date range (prevents DoS with extreme dates)
        if dt.year < 2000 or dt.year > 2100:
            raise ValidationError(
                f"{param_name} must be between 2000-01-01 and 2100-12-31",
                param_name
            )
        
        return dt
    
    @staticmethod
    def validate_date_range(
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Validate a date range, ensuring start <= end.
        
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        start = InputValidator.validate_date(start_date, "start_date")
        end = InputValidator.validate_date(end_date, "end_date")
        
        if start and end and start > end:
            raise ValidationError(
                "start_date must be before or equal to end_date",
                "date_range"
            )
        
        return start, end
    
    @staticmethod
    def validate_integer(
        value: str,
        param_name: str = "value",
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        allow_none: bool = False
    ) -> Optional[int]:
        """
        Validate and parse integer input with optional range check.
        
        Args:
            value: String value to parse
            param_name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_none: Whether None/empty is allowed
            
        Returns:
            int or None
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if allow_none:
                return None
            raise ValidationError(f"{param_name} is required", param_name)
        
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            raise ValidationError(
                f"{param_name} must be an integer, got: {value}",
                param_name
            )
        
        if min_val is not None and int_val < min_val:
            raise ValidationError(
                f"{param_name} must be >= {min_val}, got: {int_val}",
                param_name
            )
        
        if max_val is not None and int_val > max_val:
            raise ValidationError(
                f"{param_name} must be <= {max_val}, got: {int_val}",
                param_name
            )
        
        return int_val
    
    @staticmethod
    def validate_float(
        value: str,
        param_name: str = "value",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_none: bool = False
    ) -> Optional[float]:
        """
        Validate and parse float input with optional range check.
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if allow_none:
                return None
            raise ValidationError(f"{param_name} is required", param_name)
        
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                f"{param_name} must be a number, got: {value}",
                param_name
            )
        
        if min_val is not None and float_val < min_val:
            raise ValidationError(
                f"{param_name} must be >= {min_val}",
                param_name
            )
        
        if max_val is not None and float_val > max_val:
            raise ValidationError(
                f"{param_name} must be <= {max_val}",
                param_name
            )
        
        return float_val
    
    @staticmethod
    def validate_pagination(
        page: Optional[str] = None,
        per_page: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Validate pagination parameters.
        
        Defaults:
            page: 1
            per_page: 100 (max 1000)
            
        Returns:
            Tuple of (page, per_page)
        """
        page_num = InputValidator.validate_integer(
            page or "1",
            param_name="page",
            min_val=1,
            max_val=10000
        )

        per_page_num = InputValidator.validate_integer(
            per_page or "100",
            param_name="per_page",
            min_val=1,
            max_val=1000
        )

        # These will never be None since we provide default values
        assert page_num is not None
        assert per_page_num is not None
        return page_num, per_page_num
    
    @staticmethod
    def validate_horizon(horizon: str) -> int:
        """
        Validate forecast horizon.

        Valid range: 1-200 (based on current data)
        """
        result = InputValidator.validate_integer(
            horizon,
            param_name="horizon",
            min_val=1,
            max_val=200
        )
        # Will never be None since allow_none defaults to False
        assert result is not None
        return result
    
    @staticmethod
    def validate_strategy(strategy: str) -> int:
        """
        Validate trading strategy.

        Valid values:
            9 = long-only
            10 = short-only
            11 = long/short
        """
        result = InputValidator.validate_integer(
            strategy,
            param_name="strategy",
            min_val=9,
            max_val=11
        )
        # Will never be None since allow_none defaults to False
        assert result is not None
        return result
    
    @staticmethod
    def validate_string(
        value: str,
        param_name: str = "value",
        min_length: int = 0,
        max_length: int = 1000,
        pattern: Optional[str] = None,
        allow_none: bool = False
    ) -> Optional[str]:
        """
        Validate string input with length and pattern checks.
        
        Args:
            value: String to validate
            param_name: Parameter name for error messages
            min_length: Minimum string length
            max_length: Maximum string length (prevents DoS)
            pattern: Optional regex pattern to match
            allow_none: Whether None/empty is allowed
        """
        if value is None or value.strip() == "":
            if allow_none:
                return None
            raise ValidationError(f"{param_name} is required", param_name)
        
        value = value.strip()
        
        if len(value) < min_length:
            raise ValidationError(
                f"{param_name} must be at least {min_length} characters",
                param_name
            )
        
        if len(value) > max_length:
            raise ValidationError(
                f"{param_name} must be at most {max_length} characters",
                param_name
            )
        
        if pattern and not re.match(pattern, value):
            raise ValidationError(
                f"{param_name} has invalid format",
                param_name
            )
        
        return value
    
    @staticmethod
    def sanitize_for_logging(value: Optional[str], max_length: int = 100) -> str:
        """
        Sanitize a value for safe logging (prevent log injection).
        
        - Truncates to max_length
        - Removes newlines and control characters
        - Masks potential secrets
        """
        if not value:
            return "<empty>"
        
        # Remove control characters and newlines
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Truncate
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        # Mask if looks like a secret
        if re.match(r'^(qdt_|sk-|api_|key_)', sanitized, re.I):
            return sanitized[:8] + "..." + sanitized[-4:]
        
        return sanitized
