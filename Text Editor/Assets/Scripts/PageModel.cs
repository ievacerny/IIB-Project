using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PageModel{

    #region Private Attributes
    private string text = "";
    private LimitedStack<string> undo_stack = new LimitedStack<string>(10);
    private LimitedStack<string> redo_stack = new LimitedStack<string>(10);
    #endregion

    #region Constructor
    public PageModel(string initial_model)
    {
        text = initial_model;
    }
    #endregion

    #region Read Methods

    /// <summary>
    /// Returns the character at index i.
    /// </summary>
    /// <param name="i"></param>
    /// <returns></returns>
    public char GetLetter(int i)
    {
        if (i < text.Length)
            return text[i];

        Debug.LogWarning(string.Format(
            "Index {0} to GetLetter out of range. Text length: {1}",
            i, text.Length));
        return '\0';
    }

    /// <summary>
    /// Returns the word at index i.
    /// </summary>
    /// <param name="i"></param>
    /// <returns></returns>
    public string GetWord(int i)
    {
        if (text[i] == ' ')
            return "";

        int start = text.LastIndexOf(' ', i);
        int end = text.IndexOf(' ', i);

        int line_start = text.LastIndexOf('\n', i);
        int line_end = text.IndexOf('\n', i);

        // Word limiting indices will be the smallest range between the 2
        start = (int)Mathf.Max(start, line_start);
        end = (int)Mathf.Min(end, line_end);

        // If start=-1, will start reading from 0
        if (end == -1)
            end = text.Length;

        int length = end - (start + 1);

        return text.Substring(start + 1, length);
    }

    /// <summary>
    /// Return the text between the specified indices (inclusive).
    /// </summary>
    /// <param name="i1"></param>
    /// <param name="i1"></param>
    /// <returns></returns>
    public string GetSelection(int i1, int i2)
    {
        if (i1 > i2)
        {
            int temp = i1; i1 = i2; i2 = temp;
        }
        else if (i1 == i2)
        {
            return text[i1].ToString();
        }

        return text.Substring(i1, i2 - i1 + 1);
    }

    /// <summary>
    /// Returns full text.
    /// </summary>
    /// <returns></returns>
    public string GetText()
    {
        return text;
    }

    #endregion

    #region Modify Methods

    /// <summary>
    /// Sets the text of the model to be new_text.
    /// </summary>
    /// <param name="new_text"></param>
    public void UpdateModel(string new_text)
    {
        text = new_text;
        undo_stack.Push(text);
    }

    public void Undo()
    {
        if (undo_stack.Count != 0)
        {
            redo_stack.Push(undo_stack.Pop());
            text = undo_stack.Peek();
        }
    }

    public void Redo()
    {
        if (redo_stack.Count != 0)
        {
            undo_stack.Push(redo_stack.Pop());
            text = undo_stack.Peek();
        }
    }

    #endregion
}
