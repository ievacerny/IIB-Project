using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;

[TestFixture]
public class TestPresenterCoordinates
{
    PageView view;

    [SetUp]
    public void Init()
    {
        GameObject page = GameObject.FindGameObjectWithTag("Page");
        view = page.GetComponent<PageView>();
    }

    [TestCase("ABCDE", "A\nB\nC\nD\nE", 2, 2, 0)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 3, 2, 0)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 2, 2, -1)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 5, 4, 0)]
    [TestCase("Hello\n\nbeautiful world", "Hello\n\nbeautiful\nworld", 10, 2, 3)]
    public void Test_IdxToInd(string text, string rend_text, int idx, int exp_row, int exp_col)
    {
        PagePresenter presenter = new PagePresenter(view, text);
        view.UpdateRenderedText(rend_text);
        Indices actual = presenter.IdxToInd(idx);
        Assert.AreEqual(exp_row, actual.row, "Incorrect row");
        Assert.AreEqual(exp_col, actual.col, "Incorrect column");
    }

    [TestCase("ABCDE", "A\nB\nC\nD\nE", 2, 0, 2)]
    [TestCase("ABCDE", "A\nB\nC\nD\nE", 2, 6, 2)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 2, 0, 3)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 1, 6, 1)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 7, 7, 6)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", 0, -1, -1)]
    [TestCase("AB\nCDE", "A\nB\nC\nD\nE", -1, -1, -1)]
    [TestCase("Hello\n\nbeautiful world", "Hello\n\nbeautiful\nworld", 2, 3, 10)]
    public void Test_IndtoIdx(string text, string rend_text, int row, int col, int exp_idx)
    {
        PagePresenter presenter = new PagePresenter(view, text);
        view.UpdateRenderedText(rend_text);
        int actual = presenter.IndToIdx(new Indices { row = row, col = col });
        Assert.AreEqual(exp_idx, actual);
    }
}

[TestFixture]
public class TestPresenterEvents
{
    PageView view;
    PagePresenter presenter;
    PageModel model;

    [SetUp]
    public void Init()
    {
        string model_string = (
            "Gifford Woods State Park is a state park located at " + //51, length 52
            "the base of Pico Peak in Killington, Vermont.\n" + //45, length 46
            "The wooded park provides camping, picnic, and fishing " + //53, length 54
            "facilities. " + //11, length 12
            "ThisIsAVeryLongWordThatShouldBeLineWrappedCorrectlyBut" + //53, length 54
            "We'llSeeHowItGoes" // 16, length 17
        );
        string rendered_string = (
             "Gifford Woods State Park is a state park located at \n" + //51, length 52
             "the base of Pico Peak in Killington, Vermont.\n" + //45, length 46
             "The wooded park provides camping, picnic, and fishing \n" + //53, length 54
             "facilities. \n" + //11, length 12
             "ThisIsAVeryLongWordThatShouldBeLineWrappedCorrectlyBut\n" + //53, length 54
             "We'llSeeHowItGoes" // 16, length 17
         );
        GameObject page = GameObject.FindGameObjectWithTag("Page");
        view = page.GetComponent<PageView>();
        model = new PageModel(model_string);
        presenter = new PagePresenter(view, model_string, model);
        view.ChangePresenterReference(presenter);
        view.UpdateRenderedText(rendered_string);
    }

    private Indices GetVisualCursorPosition()
    {
        foreach (Transform child in view.transform)
        {
            if (child.CompareTag("Text Cursor"))
                return view.CoordsToInd(child.localPosition);
        }
        return new Indices { row = 0, col = 0 };
    }

    private List<GameObject> GetSelectionObjects()
    {
        List<GameObject> objects = new List<GameObject>();
        foreach (Transform child in view.transform)
        {
            if (child.name == "Selection(Clone)")
                objects.Add(child.gameObject);
        }
        return objects;
    }

    [TestCase(3, 10, 162, 3, 10)] // On the text
    [TestCase(6, 10, 234, 5, 16)] // After the text
    [TestCase(1, 50, 96, 1, 44)] // After the text in line
    public void Test_Click(int loc_row, int loc_col, int exp_idx, int exp_loc_row, int exp_loc_col)
    {
        Indices loc = new Indices { row = loc_row, col = loc_col };
        presenter.Click(loc);

        // Check cursor locations
        Indices visual = GetVisualCursorPosition();
        int idx = presenter.GetCursorIndex();
        Assert.AreEqual(exp_idx, idx, "String index incorrect");
        Assert.AreEqual(exp_loc_row, visual.row, "Visual cursor row incorrect");
        // Visual cursor is inbetween letter (adding 0.5 to ind), therefore +1
        Assert.AreEqual(exp_loc_col+1, visual.col, "Visual cursor column incorrect");

        // Check that no text is selected
        List<GameObject> selection = GetSelectionObjects();
        Assert.AreEqual("", presenter.GetSelectedText(), "Selected string is not empty");
        Assert.AreEqual(0, selection.Count, "Selection drawings are in the scene");
    }
    
}
