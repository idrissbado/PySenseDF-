"""
PySenseDF Sample Datasets
=========================

Included sample datasets for testing and learning.
"""

from pathlib import Path
from ..core.dataframe import DataFrame


def get_dataset_path(name: str) -> Path:
    """Get path to a sample dataset"""
    datasets_dir = Path(__file__).parent
    return datasets_dir / f"{name}.csv"


def load_customers() -> DataFrame:
    """
    Load sample customers dataset
    
    Columns:
    - customer_id: Unique customer ID
    - name: Customer name
    - city: City (New York, Los Angeles, Chicago)
    - age: Customer age
    - income: Annual income
    - revenue: Total revenue from customer
    - purchase_count: Number of purchases
    - last_purchase_date: Date of last purchase
    - status: active or inactive
    
    Returns:
        DataFrame with 20 customer records
    """
    path = get_dataset_path("customers")
    return DataFrame.read_csv(str(path))


def load_products() -> DataFrame:
    """
    Load sample products dataset
    
    Columns:
    - product_id: Unique product ID
    - product_name: Product name
    - category: Electronics, Furniture, or Stationery
    - price: Product price
    - stock: Items in stock
    - supplier: Supplier name
    - rating: Customer rating (1-5)
    - reviews_count: Number of reviews
    
    Returns:
        DataFrame with 15 product records
    """
    path = get_dataset_path("products")
    return DataFrame.read_csv(str(path))


def load_sales() -> DataFrame:
    """
    Load sample sales/orders dataset
    
    Columns:
    - order_id: Unique order ID
    - customer_id: Customer who placed order
    - product_id: Product ordered
    - quantity: Quantity ordered
    - order_date: Date order was placed
    - ship_date: Date order was shipped
    - delivery_date: Date order was delivered
    - status: delivered, in_transit, or processing
    - total_amount: Total order amount
    
    Returns:
        DataFrame with 15 order records
    """
    path = get_dataset_path("sales")
    return DataFrame.read_csv(str(path))


def list_datasets() -> list:
    """List all available sample datasets"""
    return ["customers", "products", "sales"]


# Convenience access
__all__ = [
    "load_customers",
    "load_products", 
    "load_sales",
    "list_datasets",
    "get_dataset_path"
]
