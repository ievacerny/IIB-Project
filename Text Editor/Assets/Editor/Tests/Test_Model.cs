using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

[TestFixture]
public class TestModeReadMethods
{    
    [TestCase("Hello", 0, ExpectedResult = 'H')]
    [TestCase("Hello\n", 5, ExpectedResult = '\n')]
    public char Test_GetLetter_Success(string text, int idx) {
       
        PageModel model = new PageModel(text);
        return model.GetLetter(idx);
    }
    
    [TestCase("Hello", 6, "Index {0} to GetLetter out of range. Text length: 5", ExpectedResult = '\0')]
    [TestCase("Hello", -1, "Negative index: {0}", ExpectedResult = '\0')]
    public char Test_GetLetter_ErrorMessage(string text, int idx, string error_msg)
    {
        PageModel model = new PageModel(text);
        LogAssert.Expect(LogType.Error, string.Format(error_msg, idx));
        return model.GetLetter(idx);
    }

    [TestCase("Hello world", 2, ExpectedResult = "Hello")]
    [TestCase("Hello world", 8, ExpectedResult = "world")]
    [TestCase("Hello world", 5, ExpectedResult = "")]
    [TestCase("Hello, world", 2, ExpectedResult = "Hello,")]
    [TestCase("Hello, world", 5, ExpectedResult = "Hello,")]
    [TestCase("Hello\nworld", 5, ExpectedResult = "")]
    [TestCase("Hello beautiful world", 8, ExpectedResult = "beautiful")]
    [TestCase("Hello\nbeautiful\nworld", 8, ExpectedResult = "beautiful")]
    [TestCase("\nHello beautiful\nworld", 8, ExpectedResult = "beautiful")]
    public string Test_GetWord(string text, int idx)
    {
        PageModel model = new PageModel(text);
        return model.GetWord(idx);
    }

    [TestCase("Hello world", 1, 3, ExpectedResult = "ell")]
    [TestCase("Hello world", 1, 6, ExpectedResult = "ello w")]
    [TestCase("Hello\nworld", 1, 6, ExpectedResult = "ello\nw")]
    [TestCase("Hello world", 3, 1, ExpectedResult = "ell")]
    [TestCase("Hello world", 0, 0, ExpectedResult = "H")]
    public string Test_GetSelection(string text, int i1, int i2)
    {
        PageModel model = new PageModel(text);
        return model.GetSelection(i1, i2);
    }
    
    [TestCase("Hello world")]
    [TestCase("Hello\nworld")]
    public void Test_GetText(string text)
    {
        PageModel model = new PageModel(text);
        Assert.AreEqual(text, model.GetText());
    }
}

[TestFixture]
public class TestModelModifyMethods
{
    [Test]
    public void Test_UpdateModel()
    {
        PageModel model = new PageModel("string 1");
        model.UpdateModel("string 2");
        Assert.AreEqual("string 2", model.GetText());
    }

    [TestCase(1, 1, ExpectedResult = "string 0")]
    [TestCase(0, 1, ExpectedResult = "string 0")]
    [TestCase(3, 2, ExpectedResult = "string 1")]
    [TestCase(13, 13, ExpectedResult = "string 4")]
    public string Test_Undo(int no_updates, int no_undos)
    {
        PageModel model = new PageModel("string 0");

        for (int i=1; i<=no_updates; i++)
            model.UpdateModel(string.Format("string {0}", i));

        for (int j = 1; j <= no_undos; j++)
            model.Undo();

        return model.GetText();
    }

    [TestCase(1, 1, 1, ExpectedResult = "string 1")]
    [TestCase(1, 1, 2, ExpectedResult = "string 1")]
    [TestCase(1, 2, 1, ExpectedResult = "string 1")]
    [TestCase(13, 13, 13, ExpectedResult = "string 13")]
    public string Test_Redo(int no_updates, int no_undos, int no_redos)
    {
        PageModel model = new PageModel("string 0");

        for (int i = 1; i <= no_updates; i++)
            model.UpdateModel(string.Format("string {0}", i));

        for (int j = 1; j <= no_undos; j++)
            model.Undo();

        for (int k= 1; k <= no_undos; k++)
            model.Redo();

        return model.GetText();
    }
    
    [Test]
    public void Test_Redo_Cleared()
    {
        PageModel model = new PageModel("string 0");
        model.UpdateModel("string 1");
        model.Undo();
        model.UpdateModel("string 2");
        model.Redo();
        Assert.AreEqual("string 2", model.GetText());
    }
}
