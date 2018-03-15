package com.hankcs.hanlp.model.perceptron.cli;

import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Properties;

/**
 * 解析命令行
 */
public class PropertiesArgs
{
    /**
     * Parse properties instead of String arguments.  Any additional arguments need to be passed some other way.
     * This is often used in a second pass when the property filename is passed on the command line.  Because of
     * required properties you must be careful to set them all in the property file.
     *
     * @param target    Either an instance or a class
     * @param arguments The properties that contain the arguments
     */
    public static void parse(Object target, Properties arguments)
    {
        Class clazz;
        if (target instanceof Class)
        {
            clazz = (Class) target;
        }
        else
        {
            clazz = target.getClass();
        }
        for (Field field : clazz.getDeclaredFields())
        {
            processField(target, field, arguments);
        }
        try
        {
            BeanInfo info = Introspector.getBeanInfo(clazz);
            for (PropertyDescriptor pd : info.getPropertyDescriptors())
            {
                processProperty(target, pd, arguments);
            }
        }
        catch (IntrospectionException e)
        {
            // If its not a JavaBean we ignore it
        }
    }

    private static void processField(Object target, Field field, Properties arguments)
    {
        Argument argument = field.getAnnotation(Argument.class);
        if (argument != null)
        {
            String name = Args.getName(argument, field);
            String alias = Args.getAlias(argument);
            Class type = field.getType();
            Object value = arguments.get(name);
            if (value == null && alias != null)
            {
                value = arguments.get(alias);
            }
            if (value != null)
            {
                if (type == Boolean.TYPE || type == Boolean.class)
                {
                    value = true;
                }
                Args.setField(type, field, target, value, argument.delimiter());
            }
            else
            {
                if (argument.required())
                {
                    throw new IllegalArgumentException("You must set argument " + name);
                }
            }
        }
    }

    private static void processProperty(Object target, PropertyDescriptor property, Properties arguments)
    {
        Method writeMethod = property.getWriteMethod();
        if (writeMethod != null)
        {
            Argument argument = writeMethod.getAnnotation(Argument.class);
            if (argument != null)
            {
                String name = Args.getName(argument, property);
                String alias = Args.getAlias(argument);
                Object value = arguments.get(name);
                if (value == null && alias != null)
                {
                    value = arguments.get(alias);
                }
                if (value != null)
                {
                    Class type = property.getPropertyType();
                    if (type == Boolean.TYPE || type == Boolean.class)
                    {
                        value = true;
                    }
                    Args.setProperty(type, property, target, value, argument.delimiter());
                }
                else
                {
                    if (argument.required())
                    {
                        throw new IllegalArgumentException("You must set argument " + name);
                    }
                }
            }
        }
    }
}
