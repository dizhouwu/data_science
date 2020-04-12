package pmmldemo.pmml;

import com.alibaba.fastjson.JSONObject;
import org.springframework.web.bind.annotation.*;
import org.springframework.boot.autoconfigure.*;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.IOException;

@RestController
@EnableAutoConfiguration
public class Controller {

    @RequestMapping("/")
    public String index(){
        return "hello spring for test";
    }

    @RequestMapping(value= "/predict", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    public @ResponseBody String getModel(@RequestBody String feature) throws IOException, JAXBException, SAXException {

        JSONObject json = JSONObject.parseObject(feature);

        double y = PmmlPredict.predict(json);

        return String.valueOf(y);
    }
}