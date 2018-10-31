using System;
using System.Collections.Generic;

public class LimitedStack<T>
{
    #region Fields

    private int _limit;
    private LinkedList<T> _list;

    #endregion

    #region Constructors

    public LimitedStack(int maxSize)
    {
        _limit = maxSize;
        _list = new LinkedList<T>();
    }

    #endregion

    #region Public Stack Implementation

    public void Push(T value)
    {
        if (_list.Count == _limit)
            _list.RemoveLast();

        _list.AddFirst(value);
    }

    public T Pop()
    {
        if (_list.Count > 0)
        {
            T value = _list.First.Value;
            _list.RemoveFirst();
            return value;
        }
        else
        {
            throw new InvalidOperationException("The Stack is empty");
        }
    }

    public T Peek()
    {
        if (_list.Count > 0)
        {
            T value = _list.First.Value;
            return value;
        }
        else
        {
            throw new InvalidOperationException("The Stack is empty");
        }
    }

    public void Clear()
    {
        _list.Clear();
    }

    public int Count
    {
        get { return _list.Count; }
    }

    #endregion
}
