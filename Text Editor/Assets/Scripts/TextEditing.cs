using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;

public class TextEditing : MonoBehaviour
{
    // TODO: Do geometry calculations to calculate this automatically
    public int max_rows = 15;
    public int max_columns = 53;

    private TextMesh full_text;
    private ActivePage page;

    private Indices selected_text_ind;
    private string selected_text =  "";
    public int cursor_index = 0;
    private string old_text = "";

    private Model model;

    void Start()
    {
        full_text = GetComponent<TextMesh>();
        Debug.Log(full_text.text.Length);
        page = transform.GetComponentInParent<ActivePage>();
        model = new Model(full_text.text, max_rows, max_columns);
        page.model = model;
        cursor_index = model.IndicesToStringIndex(model.GetLastCharacterIndex()) + 1;
    }

    void Update()
    {
        if (page.inEditMode)
        {
            if ((Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) && Input.GetKeyDown(KeyCode.V))
            {
                string new_text = full_text.text;
                string copied_text = GUIUtility.systemCopyBuffer;
                new_text = new_text.Substring(0, cursor_index) + copied_text + new_text.Substring(cursor_index);
                cursor_index += copied_text.Length;
                // Deselect previously selected text
                DeselectText();
                page.RemoveSelection();
                UpdateCursorIndex(model.GetLastCharacterIndex());
                UpdateText(new_text);
                return; // Return so that V is not entered as part of the text
            }

            if ((Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) && Input.GetKeyDown(KeyCode.C))
            {
                if (selected_text == "")
                    Debug.Log("No text selected");
                else
                    GUIUtility.systemCopyBuffer = selected_text;
                return; // Return so that C is not entered as part of the text
            }

            if ((Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) && Input.GetKeyDown(KeyCode.Z))
            {
                UndoText();
                return; // Return so that Z is not entered as part of the text
            }

            if (Input.inputString.Length != 0)
            {
                string new_text = full_text.text;

                foreach (char c in Input.inputString)
                {
                    // \b, \n and \r are the only supported characters in inputString
                    if (c == '\b') // has backspace/delete been pressed?
                    {
                        if (new_text.Length != 0)
                        {
                            new_text = new_text.Substring(0, cursor_index - 1) + new_text.Substring(cursor_index);
                            cursor_index -= 1;
                            Vector3 cursor_coords = page.text_cursor.localPosition;
                            cursor_coords.x -= page.width;
                            page.text_cursor.localPosition = cursor_coords;
                        }
                        if (selected_text != "")
                        {
                            new_text = DeleteSelectedText();
                            DeselectText();
                            page.RemoveSelection();
                            UpdateCursorIndex(model.GetLastCharacterIndex());
                            break;
                        }
                    }
                    else if ((c == '\n') || (c == '\r')) // enter/return
                    {
                        new_text = new_text + '\n';
                    }
                    else
                    {
                        new_text = new_text.Substring(0, cursor_index) + c + new_text.Substring(cursor_index);
                        cursor_index += 1;

                        Vector3 cursor_coords = page.text_cursor.localPosition;
                        cursor_coords.x += page.width;
                        page.text_cursor.localPosition = cursor_coords;
                    }
                }

                UpdateText(new_text);
            }
        }
    }

    private string WrapLines(string text)
    {
        string[] original_lines = text.Split(new char[] {'\n'});

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
                    wrapped_lines_list.Add(current_line.Substring(0, space_index+1));
                    current_line = current_line.Substring(space_index+1);
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

    public void SelectText(Indices start, Indices end)
    {
        selected_text = model.GetSelection(start, end);
        selected_text_ind = start;
        Debug.Log(selected_text);
    }

    public void DeselectText()
    {
        selected_text = "";
        selected_text_ind = null;
        Debug.Log("Deselected");
    }

    private string DeleteSelectedText()
    {
        int index = model.IndicesToStringIndex(selected_text_ind);
        return full_text.text.Remove(index, selected_text.Length);
    }

    public void UpdateCursorIndex(Indices c)
    {
        cursor_index = model.IndicesToStringIndex(c) + 1;
        Debug.Log(cursor_index);
    }

    public void UpdateText(string text)
    {
        old_text = full_text.text;
        full_text.text = WrapLines(text);
        model.BuildModel(full_text.text, max_rows, max_columns);
    }

    public void UndoText()
    {
        if (old_text != "")
        {
            full_text.text = WrapLines(old_text);
            model.BuildModel(full_text.text, max_rows, max_columns);
            old_text = "";
        }
    }
}