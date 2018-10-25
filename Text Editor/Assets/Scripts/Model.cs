﻿using UnityEngine;
using UnityEngine.Assertions;

public class Model {

    private char[,] model;
    private string text;

    public Model(string text, int max_rows, int max_columns)
    {
        BuildModel(text, max_rows, max_columns);
    }

    public void BuildModel(string text, int max_rows, int max_columns)
    {
        this.text = text;
        model = new char[max_rows, max_columns];
        string[] lines = text.Split(new char[] { '\n' });

        // Assume, that text will always initially be within model size.
        Assert.IsTrue(lines.Length <= model.GetLength(0), "Too many lines in text");

        for (int i = 0; i < lines.Length; i++)
        {
            Assert.IsTrue(lines[i].Length <= model.GetLength(1), string.Format("Too many characters in line {0}", i));
            for (int j = 0; j < lines[i].Length; j++)
            {
                model[i, j] = lines[i][j];
            }
        }
    }

    public char GetLetter(Indices c)
    {
        if (c.row < model.GetLength(0) && c.col < model.GetLength(1))
            return model[c.row, c.col];
        else
            return '\0';
    }

    public string GetWord(Indices c)
    {
        if ((c.row >= model.GetLength(0) && c.col >= model.GetLength(1)) || model[c.row, c.col] == ' ' || model[c.row, c.col] == '\0')
            return "";

        Indices beg = new Indices{ row = c.row, col = 0 };
 
        // Find the beginning of the string (words are only on the same line)
        for (int i=c.col; i>=0; i--)
        {
            if (model[c.row, i] == ' ')
            {
                beg.col = i + 1;
                break;
            }

        }

        // Keep reading until end of line or end of word (space)
        string word = "";
        for (int i = beg.col; i < model.GetLength(1); i++)
        {
            if (model[beg.row, i] == ' ' || model[beg.row, i] == '\0')
                break;

            word += model[beg.row, i];
        }

        return word;
    }

    public string GetSelection(Indices start, Indices end)
    {
        if (start.row >= model.GetLength(0) && start.col >= model.GetLength(1))
            return "";

        if (end.IsBefore(start))
        {
            Indices temp = start;
            start = end;
            end = temp;
        }
        
        string selection = "";
        int line_start_col = start.col;
        for (int i = start.row; i < Mathf.Min(end.row+1, model.GetLength(0)); i++)
        {
            for (int j = line_start_col; j < model.GetLength(1); j++)
            {
                if (model[i,j] != '\0')
                    selection += model[i, j];
                if (i == end.row && j == end.col)
                {
                    i = model.GetLength(0); // Trick to break out of the outer loop
                    break;
                }
            }
            if (i == end.row || i == model.GetLength(0) - 1)
                selection += '\n';
            line_start_col = 0;
        }

        return selection;
    }

    public int IndicesToStringIndex(Indices c)
    {
        Indices last_ind = GetLastCharacterIndex();
        // FIXME Maybe return -1 to indicate that this is after the last char and force the caller to
        // call the function again with the correct indices
        if (last_ind.IsBefore(c))
            c = last_ind;

        string[] lines = text.Split(new char[] { '\n' });

        string str_before = "";

        for (int i=0; i<c.row; i++)
        {
            str_before += lines[i];
            if (i != c.row)
                str_before += '\n';
        }

        // FIXME should throw an error (different from above, e.g. -2) to indicate that it's after the last
        // char in the row
        str_before += lines[c.row].Substring(0, c.col);

        return str_before.Length;
    }

    public Indices GetLastCharacterIndex()
    {
        Indices c = new Indices();

        for (int i=0; i < model.GetLength(0); i++)
        {
            if (model[i, 0] == '\0')
            {
                c.row = i - 1;
                break;
            }
        }

        for (int j = 0; j < model.GetLength(1); j++)
        {
            if (model[c.row, j] == '\0')
            {
                c.col = j - 1;
                break;
            }
        }

        return c;
    }

    public Indices GetLastCharacterInRowIndex(int row)
    {
        Indices c = new Indices();
        c.row = row;

        for (int j = 0; j < model.GetLength(1); j++)
        {
            if (model[c.row, j] == '\0')
            {
                c.col = j - 1;
                break;
            }
        }

        return c;
    }

}
