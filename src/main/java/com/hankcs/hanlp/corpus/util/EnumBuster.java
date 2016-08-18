/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-03-26 PM5:35</create-date>
 *
 * <copyright file="EnumBuster.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.util;

import sun.reflect.*;

import java.lang.reflect.*;
import java.util.*;

/**
 * 动态修改Enum的对象
 * @param <E>
 */
public class EnumBuster<E extends Enum<E>>
{
    private static final Class[] EMPTY_CLASS_ARRAY =
            new Class[0];
    private static final Object[] EMPTY_OBJECT_ARRAY =
            new Object[0];

    private static final String VALUES_FIELD = "$VALUES";
    private static final String ORDINAL_FIELD = "ordinal";

    private final ReflectionFactory reflection =
            ReflectionFactory.getReflectionFactory();

    private final Class<E> clazz;

    private final Collection<Field> switchFields;

    private final Deque<Memento> undoStack =
            new LinkedList<Memento>();

    /**
     * Construct an EnumBuster for the given enum class and keep
     * the switch statements of the classes specified in
     * switchUsers in sync with the enum values.
     */
    public EnumBuster(Class<E> clazz, Class... switchUsers)
    {
        try
        {
            this.clazz = clazz;
            switchFields = findRelatedSwitchFields(switchUsers);
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException(
                    "Could not create the class", e);
        }
    }

    /**
     * Make a new enum instance, without adding it to the values
     * array and using the default ordinal of 0.
     */
    public E make(String value)
    {
        return make(value, 0,
                    EMPTY_CLASS_ARRAY, EMPTY_OBJECT_ARRAY);
    }

    /**
     * Make a new enum instance with the given ordinal.
     */
    public E make(String value, int ordinal)
    {
        return make(value, ordinal,
                    EMPTY_CLASS_ARRAY, EMPTY_OBJECT_ARRAY);
    }

    /**
     * Make a new enum instance with the given value, ordinal and
     * additional parameters.  The additionalTypes is used to match
     * the constructor accurately.
     */
    public E make(String value, int ordinal,
                  Class[] additionalTypes, Object[] additional)
    {
        try
        {
            undoStack.push(new Memento());
            ConstructorAccessor ca = findConstructorAccessor(
                    additionalTypes, clazz);
            return constructEnum(clazz, ca, value,
                                 ordinal, additional);
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException(
                    "Could not create enum", e);
        }
    }

    /**
     * This method adds the given enum into the array
     * inside the enum class.  If the enum already
     * contains that particular value, then the value
     * is overwritten with our enum.  Otherwise it is
     * added at the end of the array.
     * <p/>
     * In addition, if there is a constant field in the
     * enum class pointing to an enum with our value,
     * then we replace that with our enum instance.
     * <p/>
     * The ordinal is either set to the existing position
     * or to the last value.
     * <p/>
     * Warning: This should probably never be called,
     * since it can cause permanent changes to the enum
     * values.  Use only in extreme conditions.
     *
     * @param e the enum to add
     */
    public void addByValue(E e)
    {
        try
        {
            undoStack.push(new Memento());
            Field valuesField = findValuesField();

            // we get the current Enum[]
            E[] values = values();
            for (int i = 0; i < values.length; i++)
            {
                E value = values[i];
                if (value.name().equals(e.name()))
                {
                    setOrdinal(e, value.ordinal());
                    values[i] = e;
                    replaceConstant(e);
                    return;
                }
            }

            // we did not find it in the existing array, thus
            // append it to the array
            E[] newValues =
                    Arrays.copyOf(values, values.length + 1);
            newValues[newValues.length - 1] = e;
            ReflectionHelper.setStaticFinalField(
                    valuesField, newValues);

            int ordinal = newValues.length - 1;
            setOrdinal(e, ordinal);
            addSwitchCase();
        }
        catch (Exception ex)
        {
            throw new IllegalArgumentException(
                    "Could not set the enum", ex);
        }
    }

    /**
     * We delete the enum from the values array and set the
     * constant pointer to null.
     *
     * @param e the enum to delete from the type.
     * @return true if the enum was found and deleted;
     * false otherwise
     */
    public boolean deleteByValue(E e)
    {
        if (e == null) throw new NullPointerException();
        try
        {
            undoStack.push(new Memento());
            // we get the current E[]
            E[] values = values();
            for (int i = 0; i < values.length; i++)
            {
                E value = values[i];
                if (value.name().equals(e.name()))
                {
                    E[] newValues =
                            Arrays.copyOf(values, values.length - 1);
                    System.arraycopy(values, i + 1, newValues, i,
                                     values.length - i - 1);
                    for (int j = i; j < newValues.length; j++)
                    {
                        setOrdinal(newValues[j], j);
                    }
                    Field valuesField = findValuesField();
                    ReflectionHelper.setStaticFinalField(
                            valuesField, newValues);
                    removeSwitchCase(i);
                    blankOutConstant(e);
                    return true;
                }
            }
        }
        catch (Exception ex)
        {
            throw new IllegalArgumentException(
                    "Could not set the enum", ex);
        }
        return false;
    }

    /**
     * Undo the state right back to the beginning when the
     * EnumBuster was created.
     */
    public void restore()
    {
        while (undo())
        {
            //
        }
    }

    /**
     * Undo the previous operation.
     */
    public boolean undo()
    {
        try
        {
            Memento memento = undoStack.poll();
            if (memento == null) return false;
            memento.undo();
            return true;
        }
        catch (Exception e)
        {
            throw new IllegalStateException("Could not undo", e);
        }
    }

    private ConstructorAccessor findConstructorAccessor(
            Class[] additionalParameterTypes,
            Class<E> clazz) throws NoSuchMethodException
    {
        Class[] parameterTypes =
                new Class[additionalParameterTypes.length + 2];
        parameterTypes[0] = String.class;
        parameterTypes[1] = int.class;
        System.arraycopy(
                additionalParameterTypes, 0,
                parameterTypes, 2,
                additionalParameterTypes.length);
        Constructor<E> cstr = clazz.getDeclaredConstructor(
                parameterTypes
        );
        return reflection.newConstructorAccessor(cstr);
    }

    private E constructEnum(Class<E> clazz,
                            ConstructorAccessor ca,
                            String value, int ordinal,
                            Object[] additional)
            throws Exception
    {
        Object[] parms = new Object[additional.length + 2];
        parms[0] = value;
        parms[1] = ordinal;
        System.arraycopy(
                additional, 0, parms, 2, additional.length);
        return clazz.cast(ca.newInstance(parms));
    }

    /**
     * The only time we ever add a new enum is at the end.
     * Thus all we need to do is expand the switch map arrays
     * by one empty slot.
     */
    private void addSwitchCase()
    {
        try
        {
            for (Field switchField : switchFields)
            {
                int[] switches = (int[]) switchField.get(null);
                switches = Arrays.copyOf(switches, switches.length + 1);
                ReflectionHelper.setStaticFinalField(
                        switchField, switches
                );
            }
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException(
                    "Could not fix switch", e);
        }
    }

    private void replaceConstant(E e)
            throws IllegalAccessException, NoSuchFieldException
    {
        Field[] fields = clazz.getDeclaredFields();
        for (Field field : fields)
        {
            if (field.getName().equals(e.name()))
            {
                ReflectionHelper.setStaticFinalField(
                        field, e
                );
            }
        }
    }


    private void blankOutConstant(E e)
            throws IllegalAccessException, NoSuchFieldException
    {
        Field[] fields = clazz.getDeclaredFields();
        for (Field field : fields)
        {
            if (field.getName().equals(e.name()))
            {
                ReflectionHelper.setStaticFinalField(
                        field, null
                );
            }
        }
    }

    private void setOrdinal(E e, int ordinal)
            throws NoSuchFieldException, IllegalAccessException
    {
        Field ordinalField = Enum.class.getDeclaredField(
                ORDINAL_FIELD);
        ordinalField.setAccessible(true);
        ordinalField.set(e, ordinal);
    }

    /**
     * Method to find the values field, set it to be accessible,
     * and return it.
     *
     * @return the values array field for the enum.
     * @throws NoSuchFieldException if the field could not be found
     */
    private Field findValuesField()
            throws NoSuchFieldException
    {
        // first we find the static final array that holds
        // the values in the enum class
        Field valuesField = null;
        try
        {
            valuesField = clazz.getDeclaredField(
                    VALUES_FIELD);
        }
        catch (NoSuchFieldException e)
        {
            Field[] fields = clazz.getDeclaredFields();
            for (Field field : fields)
            {
                if (field.getName().contains(VALUES_FIELD))
                {
                    valuesField = field;
                    break;
                }
            }
        }
        if (valuesField == null)
        {
            throw new RuntimeException("本地JVM不支持自定义词性");
        }

        // we mark it to be public
        valuesField.setAccessible(true);
        return valuesField;
    }

    public void registerSwitchClass(Class[] switchUsers)
    {
        switchFields.addAll(findRelatedSwitchFields(switchUsers));
    }

    private Collection<Field> findRelatedSwitchFields(
            Class[] switchUsers)
    {
        Collection<Field> result = new LinkedList<Field>();
        try
        {
            for (Class switchUser : switchUsers)
            {
                String name = switchUser.getName();
                int i = 0;
                while (true)
                {
                    try
                    {
                        Class suspect = Class.forName(String.format("%s$%d", name, ++i));
                        Field[] fields = suspect.getDeclaredFields();
                        for (Field field : fields)
                        {
                            String fieldName = field.getName();
                            if (fieldName.startsWith("$SwitchMap$") && fieldName.endsWith(clazz.getSimpleName()))
                            {
                                field.setAccessible(true);
                                result.add(field);
                            }
                        }
                    }
                    catch (ClassNotFoundException e)
                    {
                        break;
                    }
                }
            }
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException(
                    "Could not fix switch", e);
        }
        return result;
    }

    private void removeSwitchCase(int ordinal)
    {
        try
        {
            for (Field switchField : switchFields)
            {
                int[] switches = (int[]) switchField.get(null);
                int[] newSwitches = Arrays.copyOf(
                        switches, switches.length - 1);
                System.arraycopy(switches, ordinal + 1, newSwitches,
                                 ordinal, switches.length - ordinal - 1);
                ReflectionHelper.setStaticFinalField(
                        switchField, newSwitches
                );
            }
        }
        catch (Exception e)
        {
            throw new IllegalArgumentException(
                    "Could not fix switch", e);
        }
    }

    @SuppressWarnings("unchecked")
    private E[] values()
            throws NoSuchFieldException, IllegalAccessException
    {
        Field valuesField = findValuesField();
        return (E[]) valuesField.get(null);
    }

    private class Memento
    {
        private final E[] values;
        private final Map<Field, int[]> savedSwitchFieldValues =
                new HashMap<Field, int[]>();

        private Memento() throws IllegalAccessException
        {
            try
            {
                values = values().clone();
                for (Field switchField : switchFields)
                {
                    int[] switchArray = (int[]) switchField.get(null);
                    savedSwitchFieldValues.put(switchField,
                                               switchArray.clone());
                }
            }
            catch (Exception e)
            {
                throw new IllegalArgumentException(
                        "Could not create the class", e);
            }
        }

        private void undo() throws
                NoSuchFieldException, IllegalAccessException
        {
            Field valuesField = findValuesField();
            ReflectionHelper.setStaticFinalField(valuesField, values);

            for (int i = 0; i < values.length; i++)
            {
                setOrdinal(values[i], i);
            }

            // reset all of the constants defined inside the enum
            Map<String, E> valuesMap =
                    new HashMap<String, E>();
            for (E e : values)
            {
                valuesMap.put(e.name(), e);
            }
            Field[] constantEnumFields = clazz.getDeclaredFields();
            for (Field constantEnumField : constantEnumFields)
            {
                E en = valuesMap.get(constantEnumField.getName());
                if (en != null)
                {
                    ReflectionHelper.setStaticFinalField(
                            constantEnumField, en
                    );
                }
            }

            for (Map.Entry<Field, int[]> entry :
                    savedSwitchFieldValues.entrySet())
            {
                Field field = entry.getKey();
                int[] mappings = entry.getValue();
                ReflectionHelper.setStaticFinalField(field, mappings);
            }
        }
    }
}
