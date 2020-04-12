package pmmldemo.pmml;

import com.alibaba.fastjson.JSONObject;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;
import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public class PmmlPredict {
    // the model is global variable for springboot to load and initialize pmml

    public static Evaluator evaluator;

    // the model init func. When springboot starts, this method would be called then init evaluator above.

    public static void initModel() throws IOException, SAXException, JAXBException{

        File file = new File("C:\\Users\\BruceWDZ\\springboot_pmml\\testpmml.pmml");
        evaluator = new LoadingModelEvaluatorBuilder()
                .load(file)
                .build();
        evaluator.verify();
    }

    // define a print
    public static void print(Object ... args){
        Arrays.stream(args).forEach(System.out::print);
        System.out.println("");
    }

    // define a prediction func. The http will call it and return prediction
    // the param is a json with same fields
    public static Integer predict(JSONObject feature) throws JAXBException, SAXException, IOException {
        initModel();
        List<? extends InputField> inputFields = evaluator.getInputFields();
        print("the features of the modeil are: ", inputFields);
        List<? extends TargetField> targetFields = evaluator.getTargetFields();
        print("the target fields are: ", targetFields);

        // turn json into map format that evaluator would require
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for(InputField inputField: inputFields){
            FieldName inputName = inputField.getName();
            String name = inputName.getValue();
            Object rawValue = feature.getDoubleValue(name);
            FieldValue inputValue = inputField.prepare(rawValue);
            arguments.put(inputName, inputValue);
        }
        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        Map<String, ?> resultRecord = EvaluatorUtil.decode(results);
        Integer y = (Integer) resultRecord.get("y");
        print("the prediction result is: ");
        print(results);
        print(resultRecord);
        print(y);
        return y;

    }
}
