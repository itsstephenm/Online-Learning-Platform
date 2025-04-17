from django import template

register = template.Library()

@register.filter
def split(value, delimiter):
    """
    Returns a list from the string split by the specified delimiter.
    Usage:  {{ value|split:"," }}
    """
    if value:
        return value.split(delimiter)
    return [] 