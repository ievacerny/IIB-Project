using System.Collections.Generic;
using UnityEngine;

public class PagePresenter
{
    #region Model View Refereces
    private PageModel model;
    private PageView view;
    #endregion
    #region Private Parameters
    // TODO: Do geometry calculations to calculate this automatically
    // private readonly int max_rows = 15;
    private readonly int max_columns = 53;
    #endregion
    #region Private Attributes
    private int cursor_string_idx;
    private string selected_text = "";
    #endregion

    #region Constructor
    /// <param name="view">Reference to the view object</param>
    /// <param name="initial_model">Initial text for the model</param>
    /// <param name="test_model">Reference to the model object for testing purposes only</param>
    public PagePresenter(PageView view, string initial_model, PageModel test_model = null)
    {
        this.view = view;
        if (test_model != null)
            model = test_model;
        else
            model = new PageModel(initial_model);
        view.UpdateRenderedText(WrapLines(initial_model));
        UpdateCursorIndex(initial_model.Length);
    }
    #endregion

    #region Event Functions

    public void Click(Indices loc)
    {
        DeselectText();
        UpdateCursorIndex(IndToIdx(loc));
    }

    public void ClickDrag(Indices start, Indices end, bool moving = false)
    {
        // Negative values not allowed in selection
        start.row = Mathf.Max(start.row, 0);
        start.col = Mathf.Max(start.col, 0);
        end.row = Mathf.Max(end.row, 0);
        end.col = Mathf.Max(end.col, 0);

        if (!moving)
            SelectText(IndToIdx(start), IndToIdx(end));

        if (end.IsBefore(start))
            UpdateCursorIndex(IndToIdx(end));

        view.DrawSelection(start, end, max_columns);
    }

    public void ArrowKeys(KeyCode keycode)
    {
        DeselectText();
        switch (keycode)
        {
            case KeyCode.LeftArrow:
                UpdateCursorIndex(cursor_string_idx - 1);
                break;
            case KeyCode.RightArrow:
                UpdateCursorIndex(cursor_string_idx + 1);
                break;
            case KeyCode.UpArrow:
                Indices cur_up = IdxToInd(cursor_string_idx);
                cur_up.row -= 1;
                UpdateCursorIndex(IndToIdx(cur_up));
                break;
            case KeyCode.DownArrow:
                Indices cur_down = IdxToInd(cursor_string_idx);
                cur_down.row += 1;
                UpdateCursorIndex(IndToIdx(cur_down));
                break;
        }

    }

    public void InputString(string input_string)
    {
        foreach (char c in input_string)
        {
            string current_text = model.GetText();
            // \b, \n and \r are the only supported characters in inputString

            if (c == '\b') // backspace
            {
                if (selected_text != "")
                {
                    DeleteText(selected_text.Length);
                    DeselectText();
                }
                else if (current_text.Length != 0)
                {
                    DeleteText(1);
                }
                UpdateCursorIndex(cursor_string_idx - 1);
                continue;
            }

            // All the other characters are simply inserted (or they overwrite selection)
            string char_to_insert = c.ToString();
            if (char_to_insert == "\r")
                char_to_insert = "\n";
            
            if (selected_text != "")
            {
                DeleteText(selected_text.Length);
                DeselectText();
            }
            InsertText(char_to_insert);
            UpdateCursorIndex(cursor_string_idx + 1);
        }
    }

    public void Paste()
    {
        // Modify the text
        string copied_text = GUIUtility.systemCopyBuffer;
        InsertText(copied_text);
        // Update cursor position
        UpdateCursorIndex(cursor_string_idx + copied_text.Length);
        // Deselect previously selected text
        DeselectText();
        view.ShowTextCursor(true);
    }

    public void Copy()
    {
        if (selected_text == "")
            Debug.Log("No text selected");
        else
            GUIUtility.systemCopyBuffer = selected_text;
    }

    public void Undo()
    {
        DeselectText();
        model.Undo();
        string new_text = model.GetText();
        view.UpdateRenderedText(WrapLines(new_text));
        UpdateCursorIndex(model.GetText().Length);
    }

    public void Redo()
    {
        DeselectText();
        model.Redo();
        string new_text = model.GetText();
        view.UpdateRenderedText(WrapLines(new_text));
        UpdateCursorIndex(model.GetText().Length);
    }

    public void Reset()
    {
        DeselectText();
    }

    #endregion

    #region Private Methods

    private void UpdateCursorIndex(int new_idx)
    {
        if (new_idx < -1)
            new_idx = -1;
        else if (new_idx >= model.GetText().Length)
            new_idx = model.GetText().Length-1;

        cursor_string_idx = new_idx;
        Indices cursor_position = IdxToInd(cursor_string_idx);
        view.SetTextCursorPosition(cursor_position);
    }

    private void SelectText(int start_idx, int end_idx)
    {
        selected_text = model.GetSelection(start_idx, end_idx);
        UpdateCursorIndex(start_idx);
    }

    private void DeselectText()
    {
        selected_text = "";
        view.RemoveSelection();
        view.ShowTextCursor(true);
    }

    private void InsertText(string text_to_insert)
    {
        string new_text = model.GetText();
        // Insert after current index
        new_text = (new_text.Substring(0, cursor_string_idx+1) + text_to_insert +
                    new_text.Substring(cursor_string_idx+1));
        model.UpdateModel(new_text);
        view.UpdateRenderedText(WrapLines(new_text));
    }

    private string DeleteText(int length)
    {
        string new_text = model.GetText();
        new_text = (new_text.Substring(0, cursor_string_idx) +
                    new_text.Substring(cursor_string_idx + length));
        model.UpdateModel(new_text);
        view.UpdateRenderedText(WrapLines(new_text));
        return new_text;
    }

    private string WrapLines(string text)
    {
        string[] original_lines = text.Split(new char[] { '\n' });

        List<string> wrapped_lines_list = new List<string>();

        foreach (string line in original_lines)
        {
            if (line.Length > max_columns)
            {
                string current_line = line;
                while (current_line.Length > max_columns)
                {
                    // Find the last space before the character limit
                    int space_index = current_line.LastIndexOf(' ', max_columns);
                    if (space_index == -1)
                        space_index = max_columns - 1; // -1 to get around the space issue
                    wrapped_lines_list.Add(current_line.Substring(0, space_index + 1));
                    current_line = current_line.Substring(space_index + 1);
                }
                wrapped_lines_list.Add(current_line);
            }
            else
            {
                wrapped_lines_list.Add(line);
            }
        }

        string wrapped_lines = string.Join("\n", wrapped_lines_list.ToArray());

        return wrapped_lines;
    }

    #endregion

    #region Coordinate Conversion

    public Indices IdxToInd(int idx)
    {
        string model_text = model.GetText();
        string view_text = view.GetRenderedText();

        string[] view_lines = view_text.Split(new char[] { '\n' });

        int view_idx = 0; int line_breaks = 0;
        for (int i=0; i<view_lines.Length; i++)
        {
            if (view_idx + view_lines[i].Length > idx + line_breaks)
            {
                return new Indices {
                    row = i,
                    col = idx + line_breaks - view_idx
                };
            }

            if (model_text[view_idx + view_lines[i].Length - line_breaks] != '\n')
                line_breaks++;

            view_idx += view_lines[i].Length + 1; // +1 for \n char that is not in the line
        }
        // Not found error
        Debug.LogError("Indices not found");
        return new Indices { row = 0, col = 0 };
    }

    public int IndToIdx(Indices ind)
    {
        string model_text = model.GetText();
        string view_text = view.GetRenderedText();

        string[] view_lines = view_text.Split(new char[] { '\n' });

        if (ind.row >= view_lines.Length)
            return model_text.Length;
        if (ind.row < 0)
            return -1;

        int view_idx = 0; int line_breaks = 0;
        for (int i=0; i<ind.row; i++)
        {
            if (model_text[view_idx + view_lines[i].Length - line_breaks] != '\n')
                line_breaks++;

            view_idx += view_lines[i].Length + 1; // +1 for \n char that is not in the line
        }
        view_idx += Mathf.Min(ind.col, view_lines[ind.row].Length - 1);

        return view_idx - line_breaks;
    }

    #endregion

    #region Testing Interface

    public int GetCursorIndex()
    {
        return cursor_string_idx;
    }

    public string GetSelectedText()
    {
        return selected_text;
    }

    #endregion
}
